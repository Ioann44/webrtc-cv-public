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
let scene, camera, renderer, model, pointMesh;
const halfPI = Math.PI / 2;
const vecX = new THREE.Vector3(1, 0, 0), vecY = new THREE.Vector3(0, 1, 0), vecZ = new THREE.Vector3(0, 0, 1);
var reserveQuaternion = new THREE.Quaternion();

function applyEuler(bone, rotX, rotY, rotZ) {
    let oldParent = bone.parent;
    scene.attach(bone);

    reserveQuaternion.setFromAxisAngle(vecX, rotX);
    bone.quaternion.copy(reserveQuaternion);
    reserveQuaternion.setFromAxisAngle(vecY, rotY);
    bone.applyQuaternion(reserveQuaternion);
    reserveQuaternion.setFromAxisAngle(vecZ, rotZ);
    bone.applyQuaternion(reserveQuaternion);

    oldParent.attach(bone);
}

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
        select.removeChild(select.lastChild);
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
            // dataChannelLog.textContent += '> ' + message + '\n';
            dc.send(message);
        }, 20);
    });

    dc.addEventListener('message', (evt) => {
        // Логируем частоту обновления
        // const currentTime = performance.now();
        // const deltaTime = currentTime - lastUpdateTime;
        // lastUpdateTime = currentTime;
        // console.log(`Update interval: ${deltaTime} ms`);
        try {
            const data = JSON.parse(evt.data);
            if (data.body) {
                updateAvatarPose(data);
            }
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
    const device = document.getElementById('video-input').value;
    if (device) {
        videoConstraints.deviceId = { exact: device };
    } else {
        // alert('Please select a video input device.')
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
        // camera.position.set(0, 1.5, -3);
        // camera.rotation.y = Math.PI;

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
            // model.getObjectByName("Bip01").rotateY(Math.PI);
            // model.getObjectByName("Bip01_Spine4").rotateX(Math.PI);

            const skeletonHelper = new THREE.SkeletonHelper(model);
            scene.add(skeletonHelper);
            const axesHelper = new THREE.AxesHelper(5);
            scene.add(axesHelper);
            const pointGeometry = new THREE.SphereGeometry(0.05);
            const pointMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            pointMesh = new THREE.Mesh(pointGeometry, pointMaterial);
            pointMesh.position.copy(new THREE.Vector3(-2, 2.5, 0));
            scene.add(pointMesh);

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

let lastUpdateTime = performance.now();
// const parts = [
//     "Bip01_Head1", "Bip01_Spine4", "Bip01_Spine2", "Bip01_L_UpperArm", "Bip01_R_UpperArm",
//     "Bip01_L_Forearm", "Bip01_R_Forearm", "Bip01_L_Hand", "Bip01_R_Hand", "Bip01_L_Thigh", "Bip01_R_Thigh",
//     "Bip01_L_Calf", "Bip01_R_Calf", "Bip01_L_Foot", "Bip01_R_Foot"
// ];
const parts = [
    "Bip01_Pelvis", "Bip01_Head1", "Bip01_Spine2", "Bip01_L_UpperArm", "Bip01_R_UpperArm",
    "Bip01_L_Forearm", "Bip01_R_Forearm", "Bip01_L_Hand", "Bip01_R_Hand",
    "Bip01_L_Thigh", "Bip01_R_Thigh", "Bip01_L_Calf", "Bip01_R_Calf", "Bip01_L_Foot", "Bip01_R_Foot"
];

const pos_mods = { x: [2, 0], y: [-2, 1.5], z: [-2, 0] }
const quarter = Math.PI / 4;

function updateAvatarPose(data) {
    if (!model || !data.body) return;
    delete data.body.Bip01_Head1.position;
    // delete data.body.Bip01_Head1.rotation;
    // delete data.body.Bip01_Spine2.position;
    // delete data.body.Bip01_Spine2.rotation;

    parts.forEach(part => {
        const bone = model.getObjectByName(part);
        if (bone && data.body[part]) {
            if (data.body[part].position) {
                const position = data.body[part].position;
                const globalPos = new THREE.Vector3(
                    position[0] * pos_mods.x[0] + pos_mods.x[1],
                    position[1] * pos_mods.y[0] + pos_mods.y[1],
                    position[2] * pos_mods.z[0] + pos_mods.z[1]
                );

                if (bone.parent.isBone) {
                    if (bone.parent.children.length === 1) {
                        bone.parent.lookAt(globalPos);
                        bone.parent.rotateY(-halfPI);
                    }
                }

                bone.parent.worldToLocal(globalPos);
                bone.position.copy(globalPos);
            }
            if (data.body[part].rotation) {
                // yaw, roll, pitch
                const rotation = data.body[part].rotation;

                // I hate gimbal lock
                if (part === "Bip01_Spine2") {
                    applyEuler(bone, rotation[0] + halfPI, -rotation[2], rotation[1] + halfPI);
                }
                if (part === "Bip01_Head1") {
                    applyEuler(bone, rotation[0] - halfPI, rotation[2], rotation[1] + halfPI)
                }
                if (part === "Bip01_Pelvis") {
                    applyEuler(bone, rotation[2], rotation[0], rotation[1]);
                }
            }
        }
    });

    model.updateMatrixWorld(true);

    // Логируем частоту обновления
    // const currentTime = performance.now();
    // const deltaTime = currentTime - lastUpdateTime;
    // lastUpdateTime = currentTime;
    // console.log(`Update interval: ${deltaTime} ms`);
}


document.getElementById('start').addEventListener('click', start)

enumerateInputDevices();

start()