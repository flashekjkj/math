"""内置网页聊天界面（单文件，无前端依赖）。

由 / 路由返回；通过 /v1/chat/completions 的 stream 接口边生成边渲染，
同时展示模型/显存信息与本次生成的首字延迟、速度和 token 数。
"""

PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Math Tutor · 数学辅导演示</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, "Microsoft YaHei", sans-serif;
         background: #0f1115; color: #e6e8ee; }
  header { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
           padding: 10px 20px; border-bottom: 1px solid #262a33;
           background: #14171d; position: sticky; top: 0; z-index: 2; }
  header h1 { font-size: 16px; margin: 0; }
  #model-badge { font-size: 12px; color: #9aa3b2; }
  header select, header button { background: #1d2129; color: #e6e8ee;
           border: 1px solid #333a46; border-radius: 6px; padding: 5px 10px;
           font-size: 13px; cursor: pointer; }
  #chat { max-width: 840px; margin: 0 auto; padding: 16px 16px 120px;
          display: flex; flex-direction: column; gap: 14px; }
  .msg { max-width: 86%; padding: 10px 14px; border-radius: 12px;
         white-space: pre-wrap; word-break: break-word; line-height: 1.6;
         font-size: 15px; }
  .msg.user { align-self: flex-end; background: #2b3a6e; }
  .msg.assistant { align-self: flex-start; background: #1a1f27;
                   border: 1px solid #2a303a; }
  .meta { font-size: 11px; color: #7b8594; margin: 3px 4px 0; }
  .cursor { animation: blink 1s steps(1) infinite; color: #6c8cff; }
  @keyframes blink { 50% { opacity: 0; } }
  footer { position: fixed; bottom: 0; left: 0; right: 0;
           background: #14171d; border-top: 1px solid #262a33; padding: 12px 16px; }
  #input-row { max-width: 840px; margin: 0 auto; display: flex; gap: 10px; }
  #input { flex: 1; resize: none; height: 44px; background: #1d2129;
           color: #e6e8ee; border: 1px solid #333a46; border-radius: 10px;
           padding: 10px 12px; font-size: 15px; font-family: inherit; }
  #send { width: 88px; border: none; border-radius: 10px; background: #4c6ef5;
          color: white; font-size: 15px; cursor: pointer; }
  #send:disabled { background: #2c3652; color: #8b94a8; cursor: default; }
  #status { font-size: 12px; color: #7b8594; padding: 4px 20px 0; text-align: center; }
</style>
</head>
<body>
<header>
  <h1>Math Tutor</h1>
  <span id="model-badge">加载中…</span>
  <span style="flex:1"></span>
  <label for="mode" style="font-size:13px;color:#9aa3b2">提示词模式</label>
  <select id="mode">
    <option value="solve">解题模式</option>
    <option value="tutor">辅导模式</option>
  </select>
  <button id="reset">清空对话</button>
</header>
<div id="chat"></div>
<footer>
  <div id="status"></div>
  <div id="input-row">
    <textarea id="input" placeholder="输入数学题，Enter 发送，Shift+Enter 换行"></textarea>
    <button id="send">发送</button>
  </div>
</footer>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
const modeEl = document.getElementById('mode');
let history = [];
let prompts = { solve: '', tutor: '' };

fetch('/health').then(r => r.json()).then(h => {
  document.getElementById('model-badge').textContent =
    `${h.model}${h.adapter ? ' + ' + h.adapter : ''} · ${h.device} · 显存 ${h.vram_allocated_gb}/${h.vram_total_gb} GB`;
}).catch(() => {});
fetch('/api/prompts').then(r => r.json()).then(p => { prompts = p; }).catch(() => {});

function bubble(role) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
function addMeta(text) {
  const div = document.createElement('div');
  div.className = 'meta';
  div.textContent = text;
  chat.appendChild(div);
}

document.getElementById('reset').onclick = () => { history = []; chat.innerHTML = ''; };

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  bubble('user').textContent = text;
  history.push({ role: 'user', content: text });
  const messages = [{ role: 'system', content: prompts[modeEl.value] || '' }, ...history];
  sendBtn.disabled = true;
  const out = bubble('assistant');
  out.innerHTML = '<span class="cursor">▋</span>';
  const t0 = performance.now();
  let firstTokenAt = null, content = '';
  try {
    const res = await fetch('/v1/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'math-tutor', messages, stream: true, max_tokens: 512, temperature: 0.7 })
    });
    if (!res.ok) throw new Error('HTTP ' + res.status + ' ' + await res.text());
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        for (const line of chunk.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') continue;
          const delta = JSON.parse(data).choices?.[0]?.delta?.content || '';
          if (delta) {
            if (firstTokenAt === null) firstTokenAt = performance.now();
            content += delta;
            out.textContent = content;
            chat.scrollTop = chat.scrollHeight;
          }
        }
      }
    }
    history.push({ role: 'assistant', content });
    const elapsed = (performance.now() - t0) / 1000;
    const genSec = (performance.now() - firstTokenAt) / 1000;
    addMeta(`首字 ${firstTokenAt === null ? '-' : ((firstTokenAt - t0) / 1000).toFixed(2) + 's'} · ` +
            `共 ${elapsed.toFixed(2)}s · ` +
            (genSec > 0 ? (content.length / genSec / 2).toFixed(1) + ' 字/s' : ''));
  } catch (e) {
    out.textContent = '出错了：' + e.message;
  } finally {
    out.querySelector('.cursor')?.remove();
    sendBtn.disabled = false;
    input.focus();
    statusEl.textContent = '';
  }
}

sendBtn.onclick = send;
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
</script>
</body>
</html>
"""
