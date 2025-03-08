import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, cast

import cv2
import numpy as np
import tqdm
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaRelay
from av import VideoFrame
from cv2.typing import MatLike
from module import gesture, skeleton

ROOT = os.path.dirname(__file__)
IMAGES_BUFFER_SIZE = 1

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

progress_bar = tqdm.tqdm(unit="frame", smoothing=0.1)


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.images_buffer: deque[MatLike] = deque(maxlen=IMAGES_BUFFER_SIZE)

        self.transform: Optional[str] = None
        self.is_mirrored = True

        self.processing = None
        self.last_hands_drawed = None
        self.last_detected = []
        self.last_tracked = {}
        self.avatar_data = None

    async def recv(self):
        if self.transform not in ("detection", "tracking", "avatar"):
            progress_bar.update()

        frame: VideoFrame = await self.track.recv()
        img: MatLike = frame.to_ndarray(format="bgr24")
        if self.is_mirrored:
            cv2.flip(img, 1, img)

        match self.transform:
            case "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            case "edges":
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            # case "skeleton":  # deprecated
            #     skeleton.draw_skeleton(img)
            case "hands" | "detection" | "tracking":
                self.images_buffer.append(img)
                if self.processing is None or self.processing.done():
                    progress_bar.update()
                    if self.transform == "hands":
                        if self.processing is not None:
                            self.last_hands_drawed = self.processing.result()
                        self.processing = self.executor.submit(skeleton.draw_hands, img)
                    elif self.transform == "detection":
                        if self.processing is not None:
                            self.last_detected = self.processing.result()
                        self.processing = self.executor.submit(gesture.detect, img)
                    else:
                        if self.processing is not None:
                            self.last_tracked = self.processing.result()
                        self.processing = self.executor.submit(gesture.track_objects, img)

                img = self.images_buffer[0]
                if self.transform == "hands":
                    if isinstance(self.last_hands_drawed, np.ndarray):
                        img = self.last_hands_drawed
                elif self.transform == "detection":
                    assert isinstance(self.last_detected, list)
                    gesture.draw_detections(img, self.last_detected)
                else:
                    assert isinstance(self.last_tracked, dict)
                    gesture.draw_tracks(img, self.last_tracked)
            case "avatar":
                if self.processing is None or self.processing.done():
                    progress_bar.update()
                    if self.processing is not None:
                        self.avatar_data = self.processing.result()
                    self.processing = self.executor.submit(skeleton.get_avatar_coordinates, img)
            case _:
                pass

        new_frame = VideoFrame.from_ndarray(cast(Any, img), format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    if any(pc.connectionState in {"new", "connecting", "connected"} for pc in pcs):
        return web.Response(status=429)

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    logger.info("Created for %s", request.remote)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))

    # MARK: save to
    save_to = None
    # save_to = "recorded.mp4"
    recorder = save_to and MediaRecorder(save_to) or MediaBlackhole()
    last_message_time = time.time()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            nonlocal last_message_time
            last_message_time = time.time()

            if isinstance(message, str):
                video_track = pc.getSenders()[0].track
                assert isinstance(video_track, VideoTransformTrack)
                message_data = json.loads(message)
                new_transform = message_data["transform"]
                video_track.is_mirrored = message_data["mirror"]
                if video_track.transform != new_transform:
                    video_track.transform = new_transform
                    logger.info(f'Transform changed to "{new_transform}"')
                    video_track.processing = None

                if video_track.transform == "avatar" and video_track.avatar_data:
                    channel.send(json.dumps(video_track.avatar_data))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Track %s received", track.kind)
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(relay.subscribe(track)))

        @track.on("ended")
        async def on_ended():
            logger.info("Track %s ended", track.kind)
            await recorder.stop()

    async def check_connection_activity():
        nonlocal last_message_time
        while pc.connectionState in ("new", "connecting"):
            await asyncio.sleep(1)
            last_message_time = time.time()
        while pc.connectionState == "connected":
            if time.time() - last_message_time > 2:
                logger.info("Closing connection due to inactivity")
                await pc.close()
                pcs.discard(pc)
                break
            await asyncio.sleep(1)

    # Start checking connection activity
    asyncio.ensure_future(check_connection_activity())

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def is_busy(request):
    return web.Response(
        status=200,
        text="true" if any(pc.connectionState in {"new", "connecting", "connected"} for pc in pcs) else "false",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    app.router.add_post("/busy", is_busy)
    web.run_app(app, access_log=None, host="127.0.0.1", port=5000)
