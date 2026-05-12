"""Microsoft STT module for Wyoming."""

import asyncio
import contextlib
import logging

import azure.cognitiveservices.speech as speechsdk

from . import SpeechConfig

_LOGGER = logging.getLogger(__name__)


class RecognitionSession:
    """One transcription: push audio, await final transcript."""

    def __init__(
        self,
        speech_config: speechsdk.SpeechConfig,
        samples_per_second: int,
        bits_per_sample: int,
        channels: int,
        language_kwargs: dict,
    ) -> None:
        """Build the recognizer + push stream for a single utterance."""
        self._loop = asyncio.get_running_loop()
        self._stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=samples_per_second,
                bits_per_sample=bits_per_sample,
                channels=channels,
            )
        )
        audio_config = speechsdk.audio.AudioConfig(stream=self._stream)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            **language_kwargs,
        )
        self._session_stopped = asyncio.Event()
        self._last_text: str = ""
        self._error: Exception | None = None
        self._started = False
        self._finished = False

        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.session_stopped.connect(self._on_session_stopped)
        self._recognizer.canceled.connect(self._on_canceled)

    def _on_recognized(self, evt) -> None:
        _LOGGER.debug("RECOGNIZED: %s", evt.result)
        if evt.result.text:
            self._last_text = evt.result.text

    def _on_session_stopped(self, evt) -> None:
        _LOGGER.debug("SESSION STOPPED: %s", evt)
        self._loop.call_soon_threadsafe(self._session_stopped.set)

    def _on_canceled(self, evt) -> None:
        _LOGGER.debug("CANCELED: %s", evt)
        details = evt.result.cancellation_details
        if details.reason == speechsdk.CancellationReason.Error:
            self._error = RuntimeError(
                f"Azure STT canceled: {details.error_details}"
            )
        self._loop.call_soon_threadsafe(self._session_stopped.set)

    def start(self) -> None:
        """Begin recognition.

        Fire-and-forget — the SDK buffers pushed audio until the session
        is established.
        """
        self._recognizer.start_continuous_recognition_async()
        self._started = True

    def push_chunk(self, chunk: bytes) -> None:
        """Push a chunk of audio bytes into the recognition stream."""
        self._stream.write(chunk)

    async def finish(self) -> str:
        """Close the audio stream, await final transcript."""
        if self._finished:
            return self._last_text
        self._finished = True
        self._stream.close()
        await self._session_stopped.wait()
        stop_future = self._recognizer.stop_continuous_recognition_async()
        await self._loop.run_in_executor(None, stop_future.get)
        if self._error:
            raise self._error
        return self._last_text


class MicrosoftSTT:
    """Holds shared SpeechConfig and produces per-utterance sessions."""

    def __init__(self, speechconfig: SpeechConfig) -> None:
        """Initialize."""
        self.args = speechconfig

        try:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.args.subscription_key,
                region=self.args.service_region,
            )
            _LOGGER.info("Microsoft SpeechConfig initialized successfully.")
        except Exception as e:
            _LOGGER.error(f"Failed to initialize Microsoft SpeechConfig: {e}")
            raise

        self.set_profanity(self.args.profanity)

    def new_session(
        self,
        samples_per_second: int = 16000,
        bits_per_sample: int = 16,
        channels: int = 1,
        language: str | None = None,
    ) -> RecognitionSession:
        """Create a fresh RecognitionSession for a single utterance."""
        return RecognitionSession(
            speech_config=self.speech_config,
            samples_per_second=samples_per_second,
            bits_per_sample=bits_per_sample,
            channels=channels,
            language_kwargs=self.get_language(language),
        )

    async def warmup(self) -> None:
        """Open a throwaway connection to prime SDK TLS/auth caches.

        The Azure SDK caches auth state globally, so warming one
        recognizer reduces TTFB for subsequent fresh ones.
        """
        loop = asyncio.get_running_loop()
        stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(
                samples_per_second=16000, bits_per_sample=16, channels=1
            )
        )
        try:
            audio_config = speechsdk.audio.AudioConfig(stream=stream)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config,
                **self.get_language(None),
            )
            connection = speechsdk.Connection.from_recognizer(recognizer)
            connected = asyncio.Event()
            connection.connected.connect(
                lambda evt: loop.call_soon_threadsafe(connected.set)
            )
            connection.open(True)
            try:
                await asyncio.wait_for(connected.wait(), timeout=5.0)
                _LOGGER.info("Azure STT warmed up")
            except TimeoutError:
                _LOGGER.warning("Azure STT warmup did not complete in 5s")
            with contextlib.suppress(Exception):
                connection.close()
        except Exception as e:
            _LOGGER.warning("Azure STT warmup failed: %s", e)
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    def get_language(self, language: None | str) -> dict:
        """Get the language code."""
        if len(self.args.language) > 1:
            auto_detect_source_language_config = (
                speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                    languages=self.args.language
                )
            )
            return {
                "auto_detect_source_language_config": auto_detect_source_language_config
            }

        if language:
            _LOGGER.debug(f"Language set to {language}")
            return {"language": language}

        return {"language": self.args.language[0]}

    def set_profanity(self, profanity: str):
        """Set the profanity filter level."""
        if profanity == "off":
            profanity_level = speechsdk.ProfanityOption.Raw
        elif profanity == "masked":
            profanity_level = speechsdk.ProfanityOption.Masked
        elif profanity == "removed":
            profanity_level = speechsdk.ProfanityOption.Removed
        else:
            _LOGGER.error(f"Invalid profanity level: {profanity}")
            return

        self.speech_config.set_profanity(profanity_level)
        _LOGGER.debug(f"Profanity filter set to {profanity}")
