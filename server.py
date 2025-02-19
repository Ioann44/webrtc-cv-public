import asyncio
import json
import logging
import os
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import cv2
import tqdm
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaRelay
from av import VideoFrame
from cv2.typing import MatLike
from module import gesture, skeleton

ROOT = os.path.dirname(__file__)
IMAGES_BUFFER_SIZE = 5

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

progress_bar = tqdm.tqdm(unit="frame", smoothing=0.1)


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.last_detected = []
        self.last_tracked = {}
        self.processing = None
        self.images_buffer: deque[MatLike] = deque(maxlen=IMAGES_BUFFER_SIZE)

    async def recv(self):
        if self.transform not in ("detection", "tracking"):
            progress_bar.update()

        frame: VideoFrame = await self.track.recv()
        img: MatLike = frame.to_ndarray(format="bgr24")
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
            case "skeleton":
                skeleton.draw_skeleton(img)
            case "hands":
                skeleton.draw_hands(img)
            case "detection" | "tracking":
                self.images_buffer.append(img)
                if self.processing is None or self.processing.done():
                    progress_bar.update()
                    if self.transform == "detection":
                        if self.processing is not None:
                            self.last_detected = self.processing.result()
                        self.processing = self.executor.submit(gesture.detect, img)
                    else:
                        if self.processing is not None:
                            self.last_tracked = self.processing.result()
                        self.processing = self.executor.submit(gesture.track_objects, img)

                img = self.images_buffer[0]
                if self.transform == "detection":
                    assert isinstance(self.last_detected, list)
                    gesture.draw_detections(img, self.last_detected)
                else:
                    assert isinstance(self.last_tracked, dict)
                    gesture.draw_tracks(img, self.last_tracked)

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
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))

    # MARK: save to
    save_to = None
    # save_to = "recorded.mp4"
    recorder = save_to and MediaRecorder(save_to) or MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        # if track.kind == "audio":
        #     pc.addTrack(player.audio)
        #     recorder.addTrack(track)
        if track.kind == "video":
            pc.addTrack(VideoTransformTrack(relay.subscribe(track), transform=params["video_transform"]))
            # MARK: save to
            # recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    web.run_app(app, access_log=None, host="127.0.0.1", port=5000)
