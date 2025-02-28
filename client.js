import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader';
import { TextureLoader } from 'three/src/loaders/TextureLoader';

// get DOM elements
var dataChannelLog = document.getElementById('data-channel');
var iceConnectionProgress = 0,
    iceGatheringProgress = 0,
    signalingProgress = 0;

// peer connection
var pc = null;
// data channel
var dc = null, dcInterval = null;

// Three.js variables
let scene, camera, renderer, model;

function createPeerConnection() {
    var config = { sdpSemantics: 'unified-plan' };
    // config.iceServers = [{
    //     urls: ['stun:stun.l.google.com:19302', "stun:stun1.l.google.com:19302", "stun:stun2.l.google.com: 19302"]
    // },];

    // pc = new RTCPeerConnection(config);
    pc = new RTCPeerConnection();

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', () => { iceGatheringProgress++; }, false);
    pc.addEventListener('iceconnectionstatechange', () => { iceConnectionProgress++; }, false);
    pc.addEventListener('signalingstatechange', () => { signalingProgress++; }, false);

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        document.getElementById('video').srcObject = evt.streams[0];
    });

    return pc;
}

function enumerateInputDevices() {
    const populateSelect = (select, devices) => {
        let counter = 1;
        devices.forEach((device) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || ('Device #' + counter);
            select.appendChild(option);
            counter += 1;
        });
    };

    navigator.mediaDevices.enumerateDevices().then((devices) => {
        populateSelect(
            document.getElementById('video-input'),
            devices.filter((device) => device.kind == 'videoinput')
        );
    }).catch((e) => {
        alert(e);
    });
}

function negotiate() {
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        const codecName = 'VP8/90000';
        // const video_transform = document.getElementById('video-transform').value;
        const video_transform = 'avatar';
        // const video_transform = 'skeleton';

        offer.sdp = sdpFilterCodec('video', codecName, offer.sdp);
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                video_transform: video_transform
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        alert(e);
    });
}

function start() {
    for (const node of document.getElementsByClassName("hide-on-start")) {
        node.style.display = 'none';
    }

    pc = createPeerConnection();
    var time_start = null;
    const current_stamp = () => {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    };

    var parameters = { "ordered": false, "maxRetransmits": 0 };

    dc = pc.createDataChannel('chat', parameters);
    dc.addEventListener('close', () => {
        clearInterval(dcInterval);
        dataChannelLog.textContent += '- close\n';
    });
    dc.addEventListener('open', () => {
        dataChannelLog.textContent += '- open\n';
        dcInterval = setInterval(() => {
            var message = 'ping ' + current_stamp();
            dataChannelLog.textContent += '> ' + message + '\n';
            dc.send(message);
        }, 1000);
    });

    dc.addEventListener('message', (evt) => {
        try {
            const data = JSON.parse(evt.data);
            updateAvatarPose(data);
        } catch (error) {
            console.error("Ошибка парсинга JSON:", error);
        }
    });

    // Build media constraints.
    const constraints = {
        audio: false,
        video: false
    };

    const videoConstraints = {};
    // const device = document.getElementById('video-input').value;
    const device = '4cf004769094394295453e4f926fda189961186e077ccb48382dff092568e212';
    if (device) {
        videoConstraints.deviceId = { exact: device };
    } else {
        alert('Please select a video input device.')
    }

    const resolution = ""
    // const resolution = "320x240"
    // const resolution = "640x480"
    // const resolution = "960x540"
    // const resolution = "1280x720"
    if (resolution) {
        const dimensions = resolution.split('x');
        videoConstraints.width = parseInt(dimensions[0], 0);
        videoConstraints.height = parseInt(dimensions[1], 0);
    }

    constraints.video = Object.keys(videoConstraints).length ? videoConstraints : true;

    // Acquire media and start negotiation.
    if (constraints.video) {
        document.getElementById('media').style.display = 'block';
        navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
            stream.getTracks().forEach((track) => {
                pc.addTrack(track, stream);
            });
            return negotiate();
        }, (err) => {
            alert('Could not acquire media: ' + err);
        });
    } else {
        negotiate();
    }

    // Initialize Three.js scene
    initThreeJS();
}

function sdpFilterCodec(kind, codec, realSdp) {
    var allowed = []
    var rtxRegex = new RegExp('a=fmtp:(\\d+) apt=(\\d+)\r$');
    var codecRegex = new RegExp('a=rtpmap:([0-9]+) ' + escapeRegExp(codec))
    var videoRegex = new RegExp('(m=' + kind + ' .*?)( ([0-9]+))*\\s*$')

    var lines = realSdp.split('\n');

    var isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var match = lines[i].match(codecRegex);
            if (match) {
                allowed.push(parseInt(match[1]));
            }
        }
    }

    var skipRegex = 'a=(fmtp|rtcp-fb|rtpmap):([0-9]+)';
    var sdp = '';

    isKind = false;
    for (var i = 0; i < lines.length; i++) {
        if (lines[i].startsWith('m=' + kind + ' ')) {
            isKind = true;
        } else if (lines[i].startsWith('m=')) {
            isKind = false;
        }

        if (isKind) {
            var skipMatch = lines[i].match(skipRegex);
            if (skipMatch && !allowed.includes(parseInt(skipMatch[2]))) {
                continue;
            } else if (lines[i].match(videoRegex)) {
                sdp += lines[i].replace(videoRegex, '$1 ' + allowed.join(' ')) + '\n';
            } else {
                sdp += lines[i] + '\n';
            }
        } else {
            sdp += lines[i] + '\n';
        }
    }

    return sdp;
}

function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

function initThreeJS() {
    // const videoTransform = document.getElementById('video-transform').value;
    const videoTransform = 'avatar';
    if (videoTransform === 'avatar') {
        const container = document.getElementById('model-container');
        container.style.display = 'block';

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 1.5, 3);

        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);

        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(1, 1, 1);
        scene.add(light);

        const loader = new FBXLoader();
        loader.load('https://ioann44.ru/skeleton/skeleton_model.fbx', (object) => {
            model = object;
            model.scale.set(0.04, 0.04, 0.04);
            scene.add(model);

            // Загружаем текстуру
            const textureLoader = new TextureLoader();
            textureLoader.load('https://ioann44.ru/skeleton/skeleton_texture.png', (texture) => {
                model.traverse((child) => {
                    if (child.isMesh) {
                        child.material = new THREE.MeshBasicMaterial({ map: texture });
                    }
                });
            });

            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }
            animate();
        });
    }
}

function updateAvatarPose(data) {
    if (!model || !data.body || !data.body.Bip01_Head1) return;

    const head = model.getObjectByName("Bip01_Head1");
    if (head) {
        const { position, rotation } = data.body.Bip01_Head1;

        // head.position.set(position[0], position[1], position[2]);
        head.position.set(-position[1]/60 + 10, position[2]/30 + 15, -position[0]/40 + 5);
        // head.rotation.set(rotation[0], rotation[1], rotation[2]);
        head.rotation.set(rotation[1],0,0);
        // head.rotation.set(position[1], 0, 0);
        console.log(position[1]);
    }

    model.updateMatrixWorld(true);
}

document.getElementById('start').addEventListener('click', start)

enumerateInputDevices();

start()