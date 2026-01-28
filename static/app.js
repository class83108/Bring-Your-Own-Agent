const STREAM_URL = '/api/chat/stream';
const RESET_URL = '/api/chat/reset';

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const resetBtn = document.getElementById('reset-btn');

let isSending = false;

function setDisabled(disabled) {
  isSending = disabled;
  inputEl.disabled = disabled;
  sendBtn.disabled = disabled;
}

function createBubble(role, text = '') {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  div.appendChild(bubble);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function parseSSE(chunk) {
  const events = [];
  let currentEvent = { type: null, data: null };

  chunk.split('\n').forEach((line) => {
    if (line.startsWith('event:')) {
      currentEvent.type = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      currentEvent.data = line.slice(5).trim();
    } else if (line === '') {
      if (currentEvent.type) {
        events.push({ ...currentEvent });
      }
      currentEvent = { type: null, data: null };
    }
  });

  // 處理最後一個事件（若沒以空行結尾）
  if (currentEvent.type) {
    events.push(currentEvent);
  }

  return events;
}

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message || isSending) return;

  // 顯示使用者訊息
  createBubble('user', message);
  inputEl.value = '';
  setDisabled(true);

  // 建立 assistant 訊息泡泡（等待填充）
  const assistantBubble = createBubble('assistant', '');
  let buffer = '';

  try {
    const response = await fetch(STREAM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // 嘗試解析完整的 SSE 事件（以 \n\n 分隔）
      const lastDoubleNewline = buffer.lastIndexOf('\n\n');
      if (lastDoubleNewline === -1) continue;

      const complete = buffer.slice(0, lastDoubleNewline + 2);
      buffer = buffer.slice(lastDoubleNewline + 2);

      const events = parseSSE(complete);
      for (const evt of events) {
        if (evt.type === 'token') {
          assistantBubble.textContent += evt.data;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (evt.type === 'done') {
          // 完成
        } else if (evt.type === 'error') {
          const err = JSON.parse(evt.data);
          // 移除空的 assistant 泡泡，改顯示錯誤
          assistantBubble.parentElement.remove();
          createBubble('error', `錯誤 (${err.type}): ${err.message}`);
        }
      }
    }
  } catch (err) {
    assistantBubble.parentElement.remove();
    createBubble('error', `網路錯誤: ${err.message}`);
  } finally {
    setDisabled(false);
    inputEl.focus();
  }
}

// 發送按鈕點擊
sendBtn.addEventListener('click', sendMessage);

// Enter 鍵發送（Shift+Enter 換行）
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// 清除歷史
resetBtn.addEventListener('click', async () => {
  try {
    await fetch(RESET_URL, { method: 'POST' });
    messagesEl.innerHTML = '';
  } catch {
    createBubble('error', '清除歷史失敗，請稍後重試。');
  }
});
