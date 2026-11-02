"""Optional text-to-speech feedback for data collection."""

import shutil
import subprocess
import threading


class TextToSpeech:
    def __init__(
        self,
        rate: int = 150,
        volume: float = 1.0,
        *,
        backend: str = "local",
        network_interface: str | None = None,
        speaker_id: int = 0,
    ):
        self.engine = None
        self._command = None
        self._unitree_client = None
        self._speaker_id = speaker_id
        if backend == "unitree":
            self._init_unitree(network_interface)
        elif backend == "local":
            self._init_local(rate, volume)
        else:
            raise ValueError(f"unsupported text-to-speech backend: {backend}")
        self._speech_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _init_local(self, rate: int, volume: float) -> None:
        try:
            import pyttsx3

            self.engine = pyttsx3.init(driverName="espeak")
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)
        except Exception as e:
            self._command = shutil.which("spd-say")
            if self._command and self._has_physical_audio_sink():
                print(f"[Text To Speech] pyttsx3 unavailable ({e}); using spd-say")
            else:
                self._command = None
                print(f"[Text To Speech] Initialization failed: {e}; no physical audio sink")

    def _init_unitree(self, network_interface: str | None) -> None:
        if not network_interface:
            raise ValueError("Unitree speech requires a network interface")
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

            ChannelFactoryInitialize(0, network_interface)
            client = AudioClient()
            client.SetTimeout(10.0)
            client.Init()
            # Older SDK2 Python releases contain `tts_index += tts_index`, so
            # starting at one still produces a unique sequence: 2, 4, 8, ...
            client.tts_index = max(1, int(getattr(client, "tts_index", 0)))
            self._unitree_client = client
            print(f"[Text To Speech] Using Unitree G1 speaker on {network_interface}")
        except Exception as exc:
            print(f"[Text To Speech] Unitree speaker initialization failed: {exc}")

    @property
    def available(self) -> bool:
        return (
            self.engine is not None
            or self._command is not None
            or self._unitree_client is not None
        )

    @staticmethod
    def _has_physical_audio_sink() -> bool:
        """Reject PulseAudio's silent auto-null sink when using spd-say."""
        pactl = shutil.which("pactl")
        if pactl is None:
            return True
        try:
            result = subprocess.run(
                [pactl, "get-default-sink"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return True
        sink = result.stdout.strip()
        return bool(sink) and sink != "auto_null"

    def say(self, message: str, blocking: bool = False):
        if self.available:
            if blocking:
                self._say_blocking(message)
            else:
                thread = threading.Thread(target=self._say_blocking, args=(message,), daemon=True)
                thread.start()
                self._speech_thread = thread

    def _say_blocking(self, message: str):
        with self._lock:
            try:
                if self.engine is not None:
                    self.engine.say(message)
                    self.engine.runAndWait()
                elif self._unitree_client is not None:
                    code = self._unitree_client.TtsMaker(message, self._speaker_id)
                    if code != 0:
                        raise RuntimeError(f"Unitree TTS returned {code}")
                elif self._command is not None:
                    subprocess.run(
                        [self._command, "--wait", message],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=15,
                    )
            except (
                RuntimeError,
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                print(f"[Text To Speech] Speech failed: {exc}")

    def wait_for_completion(self):
        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join()

    def close(self) -> None:
        self.wait_for_completion()
        client = self._unitree_client
        if client is None:
            return
        stub = getattr(client, "_ClientBase__stub", None)
        if stub is not None:
            recv_channel = getattr(stub, "_ClientStub__recvChannel", None)
            send_channel = getattr(stub, "_ClientStub__sendChannel", None)
            if recv_channel is not None:
                recv_channel.CloseReader()
            if send_channel is not None:
                send_channel.CloseWriter()
        self._unitree_client = None

    def print_and_say(self, message: str, say: bool = True, blocking: bool = False):
        print(message)
        if say and self.available:
            self.say(message, blocking=blocking)
