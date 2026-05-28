const icons = {
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.2a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.2a1.7 1.7 0 0 0 1.5-1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.2a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8 1.7 1.7 0 0 0 1.5 1h.2a2 2 0 1 1 0 4h-.2a1.7 1.7 0 0 0-1.5 1Z"/>',
  back: '<path d="M19 12H5"/><path d="m12 19-7-7 7-7"/>',
  home: '<path d="m3 11 9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
  bulb: '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M8.5 14a6 6 0 1 1 7 0c-.8.7-1.2 1.4-1.5 2h-4c-.3-.6-.7-1.3-1.5-2Z"/>',
  wind: '<path d="M4 9h11a3 3 0 1 0-3-3"/><path d="M4 15h14a3 3 0 1 1-3 3"/><path d="M4 12h17"/>',
  flame: '<path d="M12 22c4 0 7-2.8 7-6.6 0-2.8-1.6-4.7-3.6-6.5.2 1.9-.4 3.4-1.7 4.3.1-3.4-1.8-6.3-4.9-9.2.4 3.2-.8 5-2.1 6.7A7.5 7.5 0 0 0 5 15.4C5 19.2 8 22 12 22Z"/>',
  trash: '<path d="M3 6h18"/><path d="M8 6V4h8v2"/><path d="M19 6l-1 15H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/>',
  camera: '<rect x="3" y="7" width="13" height="10" rx="2"/><path d="m16 11 5-3v8l-5-3Z"/>',
  chat: '<path d="M21 15a4 4 0 0 1-4 4H8l-5 4V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4Z"/>',
  touch: '<path d="M8 11V7a4 4 0 0 1 8 0v4"/><path d="M12 7v9"/><path d="M8 15v-2a2 2 0 0 0-4 0v3c0 4 3 7 8 7h2c4 0 7-3 7-7v-3a2 2 0 0 0-4 0v2"/><path d="M16 15v-3a2 2 0 0 0-4 0v3"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  play: '<path d="m6 4 14 8-14 8Z"/>',
  thermometer: '<path d="M14 14.8V5a2 2 0 0 0-4 0v9.8a4 4 0 1 0 4 0Z"/>'
};

const app = document.querySelector("#app");

const state = {
  screen: "recognition",
  touchOpen: false,
  roomInput: "",
  selectedRoom: "방",
  selectedDeviceType: "조명",
  sequence: [],
  rooms: ["방", "거실", "부엌"],
  devices: [
    { room: "방", type: "조명", status: "ON", active: true },
    { room: "방", type: "에어컨", status: "OFF", active: false, temp: "20°C" },
    { room: "거실", type: "조명", status: "ON", active: true },
    { room: "거실", type: "보일러", status: "ON", active: true, temp: "22°C" },
    { room: "부엌", type: "조명", status: "OFF", active: false }
  ]
};

function Icon(name, className = "icon") {
  return `<svg class="${className}" viewBox="0 0 24 24" aria-hidden="true">${icons[name] || ""}</svg>`;
}

function render() {
  app.innerHTML = state.screen === "setup" ? SetupScreen() : MainScreen();
  bindEvents();
}

function SetupScreen() {
  return `
    <section class="tablet-frame setup-page" data-screen="setup">
      <header class="setup-header">
        <h1 class="setup-title">${Icon("settings", "icon settings-mark")} 초기 설정</h1>
        <button class="back-button" data-action="go-recognition">${Icon("back", "small-icon")} 돌아가기</button>
      </header>
      <div class="setup-grid">
        <section class="setup-card room-card">
          <h2 class="panel-title">${Icon("home")} 방 추가</h2>
          <input class="setup-input" data-input="room" placeholder="방 이름 입력" value="${escapeHtml(state.roomInput)}" />
          <button class="add-button" data-action="add-room" ${state.roomInput.trim() ? "" : "disabled"}>+ 방 추가</button>
          ${RoomList()}
        </section>
        <section class="setup-card device-card-panel">
          <h2 class="panel-title">${Icon("bulb")} 가전 추가</h2>
          <div class="field-stack">
            <label class="field-label" for="roomSelect">방 선택</label>
            <select class="setup-select" id="roomSelect" data-input="selected-room">
              ${state.rooms.map((room) => `<option value="${room}" ${room === state.selectedRoom ? "selected" : ""}>${room}</option>`).join("")}
            </select>
          </div>
          <div class="field-stack">
            <span class="field-label">가전 종류</span>
            <div class="choice-row">
              ${["조명", "에어컨", "보일러"].map((type) => `
                <button class="choice-button ${state.selectedDeviceType === type ? "is-selected" : ""}" data-action="select-device-type" data-type="${type}">
                  ${Icon(deviceIcon(type))}
                </button>
              `).join("")}
            </div>
          </div>
          <button class="add-button" data-action="add-device">+ 가전 추가</button>
          ${DeviceList()}
        </section>
      </div>
    </section>
  `;
}

function RoomList() {
  return `
    <div class="list-stack">
      ${state.rooms.map((room) => `
        <article class="list-item">
          <span>${room}</span>
          <button class="delete-button" data-action="delete-room" data-room="${room}" aria-label="${room} 삭제">${Icon("trash", "small-icon")}</button>
        </article>
      `).join("")}
    </div>
  `;
}

function DeviceList() {
  return `
    <div class="list-stack device-list">
      ${state.devices.map((device, index) => `
        <article class="list-item">
          <span class="item-leading">${Icon(deviceIcon(device.type), "small-icon")} ${device.room} - ${device.type}</span>
          <button class="delete-button" data-action="delete-device" data-index="${index}" aria-label="${device.room} ${device.type} 삭제">${Icon("trash", "small-icon")}</button>
        </article>
      `).join("")}
    </div>
  `;
}

function MainScreen() {
  return `
    <section class="tablet-frame main-shell" data-screen="${state.screen}">
      ${CameraPanel()}
      <section class="right-pane">
        ${TopTabs()}
        ${state.screen === "status" ? DeviceStatusScreen() : RecognitionScreen()}
      </section>
    </section>
  `;
}

function CameraPanel() {
  return `
    <aside class="camera-panel">
      <span class="camera-led"></span>
      <div class="camera-center">
        <div class="camera-circle">${Icon("camera")}</div>
        <div class="camera-label">카메라 피드</div>
        <div class="camera-line"></div>
      </div>
    </aside>
  `;
}

function TopTabs() {
  const isStatus = state.screen === "status";
  return `
    <nav class="top-tabs" aria-label="상단 탭">
      <button class="tab-button ${state.screen === "recognition" ? "is-active" : ""}" data-action="go-recognition">${isStatus ? "문장 인식" : "인식 모드"}</button>
      <button class="tab-button ${isStatus ? "is-active status-active" : ""}" data-action="go-status">기기 상태</button>
      <button class="tab-button settings-tab ${state.screen === "setup" ? "is-active" : ""}" data-action="go-setup" aria-label="설정">${Icon("settings", "small-icon")}</button>
    </nav>
  `;
}

function RecognitionScreen() {
  if (state.touchOpen) {
    return `
      <section class="recognition-pane is-touch-open">
        ${TouchInputPanel(true)}
        ${FooterActions()}
      </section>
    `;
  }

  return `
    <section class="recognition-pane">
      <div class="sentence-panel">
        <h2 class="section-title">${Icon("chat")} 인식된 문장</h2>
        <div class="recognized-text">수화를 인식하면 여기에 표시됩니다</div>
        <div class="legend-row">
          ${Legend("장소", "#2f80ff")}
          ${Legend("가전", "#a855f7")}
          ${Legend("정도", "#ff7300")}
          ${Legend("동작", "#0ac75a")}
          ${Legend("제어", "#6b7280")}
        </div>
      </div>
      ${TouchInputPanel(false)}
      ${FooterActions()}
    </section>
  `;
}

function TouchInputPanel(open) {
  return `
    <section class="touch-panel">
      <button class="touch-header ${open ? "is-open" : ""}" data-action="toggle-touch">
        ${Icon("touch", "small-icon")} TOUCH INPUT PANEL
        <span class="touch-chevron">${Icon("chevron", "small-icon")}</span>
      </button>
      ${open ? `
        <div class="touch-body">
          ${SelectedSequence()}
          <div class="category-grid">
            ${WordGroup("장소", "#2f80ff", ["거실", "방", "부엌"])}
            ${WordGroup("가전", "#a855f7", ["에어컨", "보일러"])}
            ${WordGroup("정도", "#ff7300", ["1도", "2도", "4도"])}
            ${WordGroup("동작", "#0ac75a", ["켜다", "끄다", "조명 켜다", "조명 끄다", "온도 높이기", "온도 낮추기"])}
          </div>
        </div>
      ` : ""}
    </section>
  `;
}

function SelectedSequence() {
  const content = state.sequence.length
    ? state.sequence.map((word) => `<span class="sequence-token">${word}</span>`).join("")
    : "단어를 선택하여 문장을 구성하세요";
  return `
    <div class="selected-sequence">
      <div class="sequence-title">${Icon("chat", "micro-icon")} SELECTED SEQUENCE</div>
      <div class="sequence-box">${content}</div>
    </div>
  `;
}

function WordGroup(label, color, words) {
  return `
    <section class="word-group">
      <div class="group-label"><span class="group-mark" style="background:${color}"></span>${label}</div>
      <div class="word-list">
        ${words.map((word) => `
          <button class="word-button ${state.sequence.includes(word) ? "is-selected" : ""}" data-action="select-word" data-word="${word}">${word}</button>
        `).join("")}
      </div>
    </section>
  `;
}

function FooterActions() {
  return `
    <footer class="footer-actions">
      <button class="clear-button" data-action="clear-sequence">${Icon("trash", "small-icon")} 전체 지우기</button>
      <button class="run-button" data-action="run-command">${Icon("play", "small-icon")} 명령 실행</button>
    </footer>
  `;
}

function DeviceStatusScreen() {
  return `
    <section class="status-pane">
      <header class="system-header">
        <h2 class="system-title"><span class="system-bar"></span> SYSTEM STATUS</h2>
        <div class="status-legend">
          <span><i class="status-dot active"></i> ACTIVE</span>
          <span><i class="status-dot"></i> INACTIVE</span>
        </div>
      </header>
      <div class="room-status-grid">
        ${state.rooms.map((room) => RoomStatusSection(room)).join("")}
      </div>
      <footer class="status-footer">
        <span>ALL SYSTEMS OPERATIONAL</span>
        <span class="connected">CONNECTED</span>
      </footer>
    </section>
  `;
}

function RoomStatusSection(room) {
  const devices = state.devices.filter((device) => device.room === room);
  return `
    <section class="room-section">
      <h3 class="room-section-title">${room}</h3>
      <div class="room-device-list">
        ${devices.map((device) => DeviceCard(device)).join("")}
      </div>
    </section>
  `;
}

function DeviceCard(device) {
  return `
    <article class="status-device-card ${device.active ? "" : "is-off"}">
      <div class="device-icon-box">${Icon(deviceIcon(device.type), "small-icon")}</div>
      <div>
        <div class="device-name">${device.type}</div>
        ${device.active && device.temp ? `<div class="device-temp">${Icon("thermometer", "micro-icon")} ${device.temp}</div>` : ""}
      </div>
      <div class="device-state"><span class="status-dot ${device.active ? "active" : ""}"></span>${device.status}</div>
    </article>
  `;
}

function Legend(label, color) {
  return `<span class="legend-item"><i class="legend-dot" style="background:${color}"></i>${label}</span>`;
}

function deviceIcon(type) {
  if (type === "에어컨") return "wind";
  if (type === "보일러") return "flame";
  return "bulb";
}

function bindEvents() {
  document.querySelectorAll("[data-action]").forEach((node) => {
    node.addEventListener("click", handleAction);
  });

  const roomInput = document.querySelector('[data-input="room"]');
  if (roomInput) {
    roomInput.addEventListener("input", (event) => {
      state.roomInput = event.target.value;
      render();
      const nextInput = document.querySelector('[data-input="room"]');
      if (nextInput) {
        nextInput.focus();
        nextInput.setSelectionRange(nextInput.value.length, nextInput.value.length);
      }
    });
  }

  const selectedRoom = document.querySelector('[data-input="selected-room"]');
  if (selectedRoom) {
    selectedRoom.addEventListener("change", (event) => {
      state.selectedRoom = event.target.value;
      render();
    });
  }
}

function handleAction(event) {
  const target = event.currentTarget;
  const action = target.dataset.action;

  if (action === "go-recognition") {
    state.screen = "recognition";
    render();
  }

  if (action === "go-status") {
    state.screen = "status";
    render();
  }

  if (action === "go-setup") {
    state.screen = "setup";
    render();
  }

  if (action === "toggle-touch") {
    state.touchOpen = !state.touchOpen;
    render();
  }

  if (action === "select-word") {
    const word = target.dataset.word;
    if (!state.sequence.includes(word)) {
      state.sequence.push(word);
    }
    render();
  }

  if (action === "clear-sequence") {
    state.sequence = [];
    render();
  }

  if (action === "run-command") {
    applyCommandFromSequence();
    state.screen = "status";
    render();
  }

  if (action === "add-room") {
    const room = state.roomInput.trim();
    if (room && !state.rooms.includes(room)) {
      state.rooms.push(room);
      state.selectedRoom = room;
      state.roomInput = "";
      render();
    }
  }

  if (action === "delete-room") {
    const room = target.dataset.room;
    state.rooms = state.rooms.filter((item) => item !== room);
    state.devices = state.devices.filter((device) => device.room !== room);
    state.selectedRoom = state.rooms[0] || "";
    render();
  }

  if (action === "select-device-type") {
    state.selectedDeviceType = target.dataset.type;
    render();
  }

  if (action === "add-device") {
    if (state.selectedRoom) {
      state.devices.push({
        room: state.selectedRoom,
        type: state.selectedDeviceType,
        status: state.selectedDeviceType === "에어컨" ? "OFF" : "ON",
        active: state.selectedDeviceType !== "에어컨",
        temp: state.selectedDeviceType === "보일러" ? "22°C" : state.selectedDeviceType === "에어컨" ? "20°C" : undefined
      });
      render();
    }
  }

  if (action === "delete-device") {
    state.devices.splice(Number(target.dataset.index), 1);
    render();
  }
}

function applyCommandFromSequence() {
  const joined = state.sequence.join(" ");
  const room = state.rooms.find((item) => joined.includes(item)) || "거실";
  const type = ["조명", "에어컨", "보일러"].find((item) => joined.includes(item)) || "조명";
  const device = state.devices.find((item) => item.room === room && item.type === type);
  if (!device) return;

  if (joined.includes("끄")) {
    device.active = false;
    device.status = "OFF";
  } else if (joined.includes("켜")) {
    device.active = true;
    device.status = "ON";
    if (!device.temp) {
      device.temp = device.type === "에어컨" ? "20°C" : device.type === "보일러" ? "22°C" : device.temp;
    }
  } else if (joined.includes("온도") || joined.includes("도")) {
    device.active = true;
    device.status = "ON";
    device.temp = joined.includes("낮") ? "20°C" : "24°C";
  }
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[char]);
}

render();
