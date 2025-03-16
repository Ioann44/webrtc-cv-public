import abc
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional

from cv2.typing import MatLike


class DelayedMedia(abc.ABC):
    _last_result: Optional[MatLike] = None
    _executor: ThreadPoolExecutor
    _processing: Optional[Future[MatLike]] = None

    def __init__(self, executor):
        self._executor = executor

    @abc.abstractmethod
    def _process(self, img: MatLike) -> MatLike:
        raise NotImplementedError

    def process_image(self, img: MatLike):
        if self._processing is None or self._processing.done():
            if self._processing is not None:
                self._last_result = self._processing.result()
            self._processing = self._executor.submit(self._process, img)
            return True
        return False

    def get_result(self, default: MatLike):
        return self._last_result or default
