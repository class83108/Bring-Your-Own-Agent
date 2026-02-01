// === API URLs ===
const STREAM_URL = '/api/chat/stream';
const RESET_URL = '/api/chat/reset';
const HISTORY_URL = '/api/chat/history';
const FILES_TREE_URL = '/api/files/tree';
const FILES_CONTENT_URL = '/api/files/content';

// === Chat DOM 元素 ===
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const resetBtn = document.getElementById('reset-btn');
const loadHistoryBtn = document.getElementById('load-history-btn');

// === 檔案瀏覽 DOM 元素 ===
const togglePanelBtn = document.getElementById('toggle-panel-btn');
const filePanel = document.getElementById('file-panel');
const fileTree = document.getElementById('file-tree');
const refreshTreeBtn = document.getElementById('refresh-tree-btn');
const previewPanel = document.getElementById('preview-panel');
const previewFilename = document.getElementById('preview-filename');
const previewContent = document.getElementById('preview-content');
const closePreviewBtn = document.getElementById('close-preview-btn');
const toggleViewBtn = document.getElementById('toggle-view-btn');
const overlay = document.getElementById('overlay');

// === 狀態 ===
let isSending = false;
let modifiedFiles = new Set();
let fileDiffs = new Map(); // 儲存檔案的 diff 資訊
let isPanelVisible = false;
let isComposing = false; // 追蹤輸入法組字狀態
let isHistoryLoaded = false; // 追蹤歷史是否已載入
let currentPreviewPath = null; // 當前預覽的檔案路徑
let currentViewMode = 'file'; // 'file' 或 'diff'

// === 初始化檢查 ===
window.addEventListener('DOMContentLoaded', () => {
  console.log('Marked 載入狀態:', typeof marked !== 'undefined' ? '已載入' : '未載入');
  console.log('Highlight.js 載入狀態:', typeof hljs !== 'undefined' ? '已載入' : '未載入');

  // 自動載入聊天歷史
  loadChatHistory();
});

// ===========================================
// 聊天功能
// ===========================================

/**
 * 更新載入歷史按鈕的狀態
 */
function updateLoadHistoryButton() {
  if (isHistoryLoaded) {
    loadHistoryBtn.disabled = true;
    loadHistoryBtn.textContent = '已載入歷史';
  } else {
    loadHistoryBtn.disabled = false;
    loadHistoryBtn.textContent = '載入歷史';
  }
}

/**
 * 載入聊天歷史
 *
 * 注意：不檢查 cookie，因為 httponly cookie 無法被 JavaScript 讀取。
 * 直接呼叫 API，後端會自動從 cookie 讀取 session_id。
 */
async function loadChatHistory() {
  if (isHistoryLoaded) {
    console.log('歷史已載入，跳過重複載入');
    return;
  }

  // 設定載入中狀態
  loadHistoryBtn.disabled = true;
  loadHistoryBtn.textContent = '載入中...';

  try {
    const response = await fetch(HISTORY_URL);
    if (!response.ok) {
      console.error('載入歷史失敗:', response.status);
      loadHistoryBtn.textContent = '載入失敗';
      setTimeout(updateLoadHistoryButton, 2000);
      return;
    }

    const data = await response.json();
    const messages = data.messages || [];

    console.log('載入歷史訊息:', messages.length, '則');

    if (messages.length === 0) {
      console.log('無歷史訊息（可能是新使用者或已清除歷史）');
      loadHistoryBtn.textContent = '無歷史記錄';
      isHistoryLoaded = true;
      return;
    }

    // 顯示歷史訊息
    messages.forEach((msg) => {
      const bubble = createBubble(msg.role, '');

      if (msg.role === 'assistant') {
        // Assistant 訊息需要渲染 Markdown
        const html = renderMarkdown(msg.content);
        bubble.innerHTML = html;
      } else {
        // User 訊息直接顯示文字
        bubble.textContent = msg.content;
      }
    });

    isHistoryLoaded = true;
    updateLoadHistoryButton();
  } catch (err) {
    console.error('載入聊天歷史錯誤:', err);
    loadHistoryBtn.textContent = '載入失敗';
    setTimeout(updateLoadHistoryButton, 2000);
  }
}

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

// 將 Markdown 文字轉換為 HTML
function renderMarkdown(text) {

  if (typeof marked === 'undefined') {
    console.error('[Markdown] marked 未載入，使用純文字');
    return text.replace(/\n/g, '<br>');
  }

  try {
    console.log('[Markdown] marked 物件類型:', typeof marked);
    console.log('[Markdown] marked.parse 是否存在:', typeof marked.parse);

    // 設定 marked 選項
    if (marked.setOptions) {
      marked.setOptions({
        highlight: function(code, lang) {
          if (typeof hljs !== 'undefined' && lang) {
            try {
              const validLang = hljs.getLanguage(lang);
              if (validLang) {
                return hljs.highlight(code, { language: lang }).value;
              }
            } catch (e) {
              console.warn('[Highlight] 語法高亮失敗:', e);
            }
          }
          return code; // 回傳原始程式碼
        },
        breaks: true,
        gfm: true,
      });
    }

    // 嘗試不同的 API 呼叫方式
    let html;
    if (typeof marked.parse === 'function') {
      console.log('[Markdown] 使用 marked.parse()');
      html = marked.parse(text);
    } else if (typeof marked === 'function') {
      console.log('[Markdown] 使用 marked()');
      html = marked(text);
    } else {
      throw new Error('無法找到 marked 的渲染方法');
    }

    return html;
  } catch (e) {
    console.error('[Markdown] 渲染失敗:', e);
    return text.replace(/\n/g, '<br>');
  }
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

  if (currentEvent.type) {
    events.push(currentEvent);
  }

  return events;
}

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message || isSending) return;

  createBubble('user', message);
  inputEl.value = '';
  setDisabled(true);

  const assistantBubble = createBubble('assistant', '');
  let buffer = '';
  let accumulatedText = ''; // 累積 Assistant 回應文字

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

      const lastDoubleNewline = buffer.lastIndexOf('\n\n');
      if (lastDoubleNewline === -1) continue;

      const complete = buffer.slice(0, lastDoubleNewline + 2);
      buffer = buffer.slice(lastDoubleNewline + 2);

      const events = parseSSE(complete);
      for (const evt of events) {
        if (evt.type === 'token') {
          // JSON 解碼以正確處理換行符等特殊字元
          const decodedToken = JSON.parse(evt.data);
          accumulatedText += decodedToken;
          assistantBubble.textContent = accumulatedText;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (evt.type === 'done') {
          // 對話完成，將累積的文字轉換為 Markdown
          console.log('[Done] 收到 done 事件，accumulatedText 長度:', accumulatedText.length);
          if (accumulatedText) {
            const html = renderMarkdown(accumulatedText);
            assistantBubble.innerHTML = html;
          }
          // 重新整理檔案樹以檢查是否有修改
          if (isPanelVisible) {
            loadFileTree();
          }
        } else if (evt.type === 'error') {
          const err = JSON.parse(evt.data);
          assistantBubble.parentElement.remove();
          createBubble('error', `錯誤 (${err.type}): ${err.message}`);
        } else if (evt.type === 'file_change') {
          // 儲存檔案的 diff 資訊並標記為已修改
          const fileData = JSON.parse(evt.data);
          fileDiffs.set(fileData.path, fileData.diff);
          markFileModified(fileData.path);
        }
      }
    }

    // 串流結束後，處理剩餘的 buffer
    if (buffer.trim()) {
      const events = parseSSE(buffer);
      for (const evt of events) {
        if (evt.type === 'done') {
          if (accumulatedText) {
            const html = renderMarkdown(accumulatedText);
            assistantBubble.innerHTML = html;
            console.log('[Stream] 已從剩餘 buffer 設定 innerHTML');
          }
          if (isPanelVisible) {
            loadFileTree();
          }
        }
      }
    }
  } catch (err) {
    assistantBubble.parentElement.remove();
    createBubble('error', `網路錯誤: ${err.message}`);
  } finally {
    // 確保最後一定會嘗試渲染 Markdown（如果還是純文字狀態）
    if (accumulatedText && assistantBubble && assistantBubble.textContent === accumulatedText) {
      console.log('[Finally] 條件符合，執行 renderMarkdown');
      const html = renderMarkdown(accumulatedText);
      assistantBubble.innerHTML = html;
    }
    setDisabled(false);
    inputEl.focus();
  }
}

// ===========================================
// 檔案瀏覽功能
// ===========================================

/**
 * 切換檔案面板顯示
 */
function toggleFilePanel() {
  isPanelVisible = !isPanelVisible;
  filePanel.classList.toggle('hidden', !isPanelVisible);
  togglePanelBtn.classList.toggle('active', isPanelVisible);
  togglePanelBtn.textContent = isPanelVisible ? '隱藏檔案' : '顯示檔案';

  // 控制遮罩層（僅在移動裝置生效）
  updateOverlay();

  if (isPanelVisible) {
    loadFileTree();
  }
}

/**
 * 載入目錄結構
 */
async function loadFileTree() {
  fileTree.innerHTML = '<div class="tree-empty">載入中...</div>';

  try {
    const treeRes = await fetch(FILES_TREE_URL);
    const treeData = await treeRes.json();

    // modifiedFiles 只從 file_change 事件更新（不再從 Redis 載入）
    renderTree(treeData.tree, fileTree);
  } catch (err) {
    fileTree.innerHTML = '<div class="tree-error">載入失敗，請稍後重試</div>';
  }
}

/**
 * 渲染目錄樹
 */
function renderTree(items, container) {
  container.innerHTML = '';

  if (!items || items.length === 0) {
    container.innerHTML = '<div class="tree-empty">目錄為空</div>';
    return;
  }

  items.forEach((item) => {
    const itemEl = document.createElement('div');
    itemEl.className = `tree-item ${item.type}`;

    if (item.type === 'file' && modifiedFiles.has(item.path)) {
      itemEl.classList.add('modified');
    }

    // 圖示
    const iconEl = document.createElement('span');
    iconEl.className = 'tree-icon';
    iconEl.textContent = item.type === 'directory' ? '📁' : '📄';

    // 名稱
    const nameEl = document.createElement('span');
    nameEl.className = 'tree-name';
    nameEl.textContent = item.name;

    itemEl.appendChild(iconEl);
    itemEl.appendChild(nameEl);
    itemEl.dataset.path = item.path;
    itemEl.dataset.type = item.type;

    container.appendChild(itemEl);

    if (item.type === 'directory' && item.children) {
      const childrenEl = document.createElement('div');
      childrenEl.className = 'tree-children collapsed';
      renderTree(item.children, childrenEl);
      container.appendChild(childrenEl);

      // 目錄點擊展開/收合
      itemEl.addEventListener('click', (e) => {
        e.stopPropagation();
        const isExpanded = !childrenEl.classList.contains('collapsed');
        childrenEl.classList.toggle('collapsed');
        iconEl.textContent = isExpanded ? '📁' : '📂';
      });
    } else if (item.type === 'file') {
      // 檔案點擊預覽
      itemEl.addEventListener('click', (e) => {
        e.stopPropagation();
        loadFileContent(item.path, item.name);
      });
    }
  });
}

/**
 * 載入檔案內容並顯示預覽
 */
async function loadFileContent(path, filename) {
  previewFilename.textContent = filename;
  previewPanel.classList.remove('hidden');
  currentPreviewPath = path;

  // 更新遮罩層狀態
  updateOverlay();

  // 檢查是否有 diff 資訊
  const hasDiff = fileDiffs.has(path);

  // 顯示或隱藏切換按鈕
  if (hasDiff) {
    toggleViewBtn.classList.remove('hidden');
    // 預設顯示 diff 視圖
    currentViewMode = 'diff';
    updateToggleViewButton();
    showDiffView(path, fileDiffs.get(path));
  } else {
    toggleViewBtn.classList.add('hidden');
    currentViewMode = 'file';
    await showFileContent(path);
  }
}

/**
 * 顯示完整檔案內容
 */
async function showFileContent(path) {
  // 顯示原始檔案內容 - 需要確保清除之前的 diff 視圖
  const previewContentEl = document.querySelector('.preview-content');
  previewContentEl.className = 'preview-content'; // 重置 class，移除 diff-view
  previewContentEl.innerHTML = '<code id="preview-content"></code>'; // 重建結構

  // 重新取得 code 元素的參照
  const codeEl = document.getElementById('preview-content');
  codeEl.textContent = '載入中...';

  try {
    const res = await fetch(`${FILES_CONTENT_URL}?path=${encodeURIComponent(path)}`);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || '載入失敗');
    }

    const data = await res.json();

    codeEl.textContent = data.content;

    // 設定語言 class 並套用語法高亮
    if (data.language && data.language !== 'plaintext') {
      codeEl.className = `language-${data.language}`;
      hljs.highlightElement(codeEl);
    }
  } catch (err) {
    codeEl.textContent = `無法載入檔案: ${err.message}`;
  }
}

/**
 * 顯示 diff 視圖（輕量版本，不使用 Diff2Html）
 */
function showDiffView(path, diffText) {
  // 清空預覽內容並設定樣式
  const previewContentEl = document.querySelector('.preview-content');
  previewContentEl.innerHTML = '';
  previewContentEl.className = 'preview-content diff-view';

  // 建立 diff 容器
  const diffContainer = document.createElement('pre');
  diffContainer.className = 'simple-diff';
  previewContentEl.appendChild(diffContainer);

  try {
    // 解析並渲染 diff，追蹤行號
    const lines = diffText.split('\n');
    let oldLineNum = 0;
    let newLineNum = 0;

    const html = lines.map(line => {
      // 判斷行的類型
      if (line.startsWith('@@')) {
        // Hunk header - 解析行號
        const match = line.match(/@@ -(\d+),?\d* \+(\d+),?\d* @@/);
        if (match) {
          oldLineNum = parseInt(match[1]);
          newLineNum = parseInt(match[2]);
        }
        return `<div class="diff-hunk">${escapeHtml(line)}</div>`;
      } else if (line.startsWith('+') && !line.startsWith('+++')) {
        // 新增行
        const lineNumStr = String(newLineNum).padStart(4, ' ');
        newLineNum++;
        return `<div class="diff-add"><span class="line-num">${lineNumStr}</span>${escapeHtml(line)}</div>`;
      } else if (line.startsWith('-') && !line.startsWith('---')) {
        // 刪除行
        const lineNumStr = String(oldLineNum).padStart(4, ' ');
        oldLineNum++;
        return `<div class="diff-del"><span class="line-num">${lineNumStr}</span>${escapeHtml(line)}</div>`;
      } else if (line.startsWith('diff ') || line.startsWith('index ') ||
                 line.startsWith('--- ') || line.startsWith('+++ ')) {
        // 檔案 header
        return `<div class="diff-header">${escapeHtml(line)}</div>`;
      } else if (line.startsWith(' ')) {
        // Context 行
        const lineNumStr = String(oldLineNum).padStart(4, ' ');
        oldLineNum++;
        newLineNum++;
        return `<div class="diff-context"><span class="line-num">${lineNumStr}</span>${escapeHtml(line)}</div>`;
      } else {
        // 其他行（空行等）
        return `<div class="diff-context">${escapeHtml(line)}</div>`;
      }
    }).join('');

    diffContainer.innerHTML = html;
    console.log('[Diff] 輕量 Diff 渲染成功:', path);
  } catch (err) {
    console.error('[Diff] Diff 渲染失敗:', err);
    previewContentEl.innerHTML = `<div class="diff-error">無法渲染 diff: ${err.message}</div>`;
  }
}

/**
 * HTML 跳脫函數
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * 切換視圖模式（Diff / 完整檔案）
 */
function toggleView() {
  if (!currentPreviewPath || !fileDiffs.has(currentPreviewPath)) {
    return;
  }

  if (currentViewMode === 'diff') {
    currentViewMode = 'file';
    showFileContent(currentPreviewPath);
  } else {
    currentViewMode = 'diff';
    showDiffView(currentPreviewPath, fileDiffs.get(currentPreviewPath));
  }

  updateToggleViewButton();
}

/**
 * 更新切換視圖按鈕的文字
 */
function updateToggleViewButton() {
  if (currentViewMode === 'diff') {
    toggleViewBtn.textContent = '完整檔案';
    toggleViewBtn.title = '查看完整檔案內容';
  } else {
    toggleViewBtn.textContent = '變更';
    toggleViewBtn.title = '查看變更內容';
  }
}

/**
 * 關閉預覽面板
 */
function closePreview() {
  previewPanel.classList.add('hidden');
  currentPreviewPath = null;
  currentViewMode = 'file';

  // 更新遮罩層狀態
  updateOverlay();
}

/**
 * 更新遮罩層顯示狀態
 * 當檔案面板或預覽面板在移動裝置上開啟時顯示遮罩
 */
function updateOverlay() {
  const shouldShowOverlay = isPanelVisible || !previewPanel.classList.contains('hidden');
  overlay.classList.toggle('hidden', !shouldShowOverlay);
}

/**
 * 關閉所有面板（由遮罩層觸發）
 */
function closeAllPanels() {
  if (isPanelVisible) {
    toggleFilePanel();
  }
  if (!previewPanel.classList.contains('hidden')) {
    closePreview();
  }
}

/**
 * 標記檔案為已修改
 */
function markFileModified(path) {
  modifiedFiles.add(path);

  // 更新 UI 中對應的樹狀項目
  const treeItem = fileTree.querySelector(`[data-path="${path}"]`);
  if (treeItem) {
    treeItem.classList.add('modified');
  }
}

// ===========================================
// 事件綁定
// ===========================================

// 聊天功能
sendBtn.addEventListener('click', sendMessage);

// 追蹤輸入法組字狀態
inputEl.addEventListener('compositionstart', () => {
  isComposing = true;
});

inputEl.addEventListener('compositionend', () => {
  isComposing = false;
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
    e.preventDefault();
    sendMessage();
  }
});

loadHistoryBtn.addEventListener('click', loadChatHistory);

resetBtn.addEventListener('click', async () => {
  try {
    await fetch(RESET_URL, { method: 'POST' });
    messagesEl.innerHTML = '';
    isHistoryLoaded = false;
    updateLoadHistoryButton();
  } catch {
    createBubble('error', '清除歷史失敗，請稍後重試。');
  }
});

// 檔案瀏覽功能
togglePanelBtn.addEventListener('click', toggleFilePanel);
refreshTreeBtn.addEventListener('click', loadFileTree);
closePreviewBtn.addEventListener('click', closePreview);
toggleViewBtn.addEventListener('click', toggleView);

// 遮罩層點擊事件（關閉所有面板）
overlay.addEventListener('click', closeAllPanels);
