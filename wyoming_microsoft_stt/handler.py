"""Event handler for clients of the server."""

import argparse
import logging
import time

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .microsoft_stt import MicrosoftSTT, RecognitionSession

_LOGGER = logging.getLogger(__name__)


class MicrosoftEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        model: MicrosoftSTT,
        *args,
        **kwargs,
    ) -> None:
        """Initialize."""
        super().__init__(*args, **kwargs)

        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.model = model

        if len(self.cli_args.language) > 1:
            _LOGGER.warning(
                f"Multiple languages specified, auto-detection will be used for these languages only: {self.cli_args.language}"
            )

        self._language: str = self.cli_args.language[0]
        self._session: RecognitionSession | None = None

    async def handle_event(self, event: Event) -> bool:
        """Handle an event."""
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self._language = transcribe.language
                _LOGGER.debug("Language set to %s", transcribe.language)
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            _LOGGER.debug(
                f"Receiving audio: {start.width * 8}bit {start.rate}Hz {start.channels}ch"
            )
            self._session = self.model.new_session(
                bits_per_sample=start.width * 8,
                samples_per_second=start.rate,
                channels=start.channels,
                language=self._language,
            )
            self._session.start()
            return True

        if AudioChunk.is_type(event.type):
            if self._session is None:
                _LOGGER.warning("Got AudioChunk before AudioStart")
                return True
            chunk = AudioChunk.from_event(event)
            self._session.push_chunk(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            if self._session is None:
                _LOGGER.warning("Got AudioStop with no active session")
                await self.write_event(Transcript(text="").event())
                return False

            session = self._session
            self._session = None
            text = ""
            try:
                start_time = time.monotonic()
                text = await session.finish()
                _LOGGER.info(
                    f"Transcription completed in {time.monotonic() - start_time:.2f} seconds"
                )
            except Exception as e:
                _LOGGER.error(f"Failed to transcribe audio: {e}")

            _LOGGER.info(text)
            await self.write_event(Transcript(text=text).event())
            _LOGGER.debug("Completed request")

            self._language = self.cli_args.language[0]
            return False

        return True
