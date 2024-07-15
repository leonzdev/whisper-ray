import math
from ray import serve
from typing import Dict, Any, Union, List
from pydantic import BaseModel
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment, TranscriptionInfo
from ..common.constants import WORD, SEGMENT, DEVICE_CPU, DEVICE_GPU, FORMAT_JSON, FORMAT_VERBOSE, FORMAT_SRT, FORMAT_VTT
from ..common.types import TranscribeInput, TranslateInput
import asyncio
import io

VALID_RESPONSE_FORMAT = [
    FORMAT_JSON, FORMAT_VERBOSE, FORMAT_SRT, FORMAT_VTT
]

@serve.deployment()
class WhisperModelService:
    def __init__(self, model_name: str, device: str, cpu_threads: int = 0):
        self.model_name = model_name
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
        return self.format_transcribe_result(input, segments, info)

    def validate_transcribe_input(self, input: TranscribeInput) -> None:
        if input.response_format not in VALID_RESPONSE_FORMAT:
            raise ValueError("Invalid response format {}".format(input.response_format))
        if input.model != self.model_name:
            raise ValueError("Invalid model {}. Need to be {}".format(input.model, self.model_name))

    async def translate(self, input: TranslateInput) -> Dict[str, Any]:
        # TODO: use the same input validation as transcribe for now
        self.validate_transcribe_input(input)
        audio_data = io.BytesIO(input.file)
        options = {
            "task": "translate",
            "temperature": [input.temperature],
            "initial_prompt": input.prompt,
        }
        segments_generator, info = self.model.transcribe(audio_data, **options)
        segments: List[Segment] = []
        for s in segments_generator:
            await asyncio.sleep(0)
            segments.append(s)
        return self.format_translate_result(input, segments, info)

    @staticmethod
    def format_transcribe_result(input: TranscribeInput, segments: List[Segment], info: TranscriptionInfo) -> Any :
        options = info.transcription_options
        if input.response_format == FORMAT_VERBOSE:
            result = {
                "task": "transcribe",
                "language": info.language,
                "duration": info.duration,
                "text": " ".join([segment.text for segment in segments]),
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
        elif input.response_format == FORMAT_SRT:
            return __class__.format_srt(segments)
        elif input.response_format == FORMAT_VTT:
            return __class__.format_vtt(segments)
        elif input.response_format == FORMAT_JSON:
            return {"text": " ".join([segment.text for segment in segments])}
        else:
            raise ValueError("Invalid response format {}".format(input.response_format))


    @staticmethod
    def format_translate_result(input: TranscribeInput, segments: List[Segment], info: TranscriptionInfo) -> Any :
        options = info.transcription_options
        if input.response_format == FORMAT_VERBOSE:
            result = {
                "task": "translate",
                "language": info.language,
                "duration": info.duration,
                "text": " ".join([segment.text for segment in segments]),
                "segments": []
            }
            for segment in segments:
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
        elif input.response_format == FORMAT_SRT:
            return __class__.format_srt(segments)
        elif input.response_format == FORMAT_VTT:
            return __class__.format_vtt(segments)
        elif input.response_format == FORMAT_JSON:
            return {"text": " ".join([segment.text for segment in segments])}
        else:
            raise ValueError("Invalid response format {}".format(input.response_format))
        
    @staticmethod
    def format_timestamp_for_srt(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = math.floor((seconds % 1) * 1000)
        output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"
        return output

    @staticmethod
    def format_srt(segments: List[Segment]) -> str:
        # see https://github.com/SYSTRAN/faster-whisper/discussions/93
        result = ""
        count = 0
        for segment in segments:
            count +=1
            duration = f"{__class__.format_timestamp_for_srt(segment.start)} --> {__class__.format_timestamp_for_srt(segment.end)}\n"
            text = f"{segment.text.lstrip()}\n\n"
            
            result += f"{count}\n{duration}{text}"  # Write formatted string to the file
        return result

    @staticmethod
    def format_timestamp_for_vtt(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = math.floor((seconds % 1) * 1000)
        output = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"
        return output

    @staticmethod
    def format_vtt(segments: List[Segment]) -> str:
        result = "WEBVTT\n\n"
        for segment in segments:
            duration = f"{__class__.format_timestamp_for_vtt(segment.start)} --> {__class__.format_timestamp_for_vtt(segment.end)}\n"
            text = f"{segment.text.lstrip()}\n\n"
            
            result += f"{duration}{text}"  # Write formatted string to the file
        return result
