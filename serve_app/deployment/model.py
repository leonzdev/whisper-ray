from ray import serve
from typing import Dict, Any, Union, List
from pydantic import BaseModel
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from ..common.constants import WORD, SEGMENT, DEVICE_CPU, DEVICE_GPU, FORMAT_JSON, FORMAT_VERBOSE
from ..common.types import TranscribeInput
import asyncio
import io

VALID_RESPONSE_FORMAT = [
    FORMAT_JSON, FORMAT_VERBOSE
]

@serve.deployment()
class WhisperModelService:
    def __init__(self, model_name: str, device: str, cpu_threads: int = 0):
        if DEVICE_CPU == device:
            self.model = WhisperModel(model_name, DEVICE_CPU, cpu_threads=cpu_threads)
        elif DEVICE_GPU == device:
            self.model = WhisperModel(model_name, DEVICE_GPU, cpu_threads=cpu_threads)
        else:
            raise ValueError('Invalid device {}'.format(device)) 

    async def transcribe(self, input: TranscribeInput) -> Union[Dict[str, Any], str]:
        self.validate_transcribe_input(input)
        # Process the audio file from memory
        audio_data = io.BytesIO(input.file)

        # Set up transcription options
        options = {
            "task": "transcribe",
            "temperature": [input.temperature],
            "initial_prompt": input.prompt,
        }
        if input.timestamp_granularities is not None and WORD in input.timestamp_granularities:
            options["word_timestamps"] = True
        segments_generator, info = self.model.transcribe(audio_data, **options)
        segments: List[Segment] = []
        for s in segments_generator:
            await asyncio.sleep(0)
            segments.append(s)
        return self.format_transcrbe_result(input, segments, info)

    @staticmethod
    def validate_transcribe_input(input: TranscribeInput) -> None:
        if input.response_format not in VALID_RESPONSE_FORMAT:
            raise ValueError("Invalid response format {}".format(input.response_format))

    @staticmethod
    def format_transcrbe_result(input: TranscribeInput, segments: List[Segment], info: TranscriptionInfo) -> Any :
        options = info.transcription_options
        if input.response_format == FORMAT_VERBOSE:
            result = {
                "text": " ".join([segment.text for segment in segments]),
                "language": info.language,
                "duration": info.duration,
            }
            if input.timestamp_granularities is not None and WORD in input.timestamp_granularities:
                result["words"] = []
            if input.timestamp_granularities is not None and SEGMENT in input.timestamp_granularities:
                result["segments"] = []
            for segment in segments:
                if hasattr(segment, 'words') and input.timestamp_granularities is not None and WORD in input.timestamp_granularities:
                    for word in segment.words:
                        result["words"].append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                        })
                if input.timestamp_granularities is not None and SEGMENT in input.timestamp_granularities:
                    result['segments'].append({
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "temperature": segment.temperature,
                    "avg_logprob": segment.avg_logprob,
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob,
                })
            return result
        # TODO: add support for srt and vtt
        elif input.response_format == FORMAT_JSON:
            return {"text": " ".join([segment.text for segment in segments])}
        else:
            raise ValueError("Invalid response format {}".format(input.response_format))
