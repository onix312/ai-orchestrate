// Behavioral tests of the dashboard controller in a minimal DOM (no external packages).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('ai_orchestrate/static/index.html', 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
new vm.Script(script); // Parse all code, including handlers below the controller.
const controller = script.slice(0, script.indexOf('    document.querySelectorAll(".mode-option").forEach(node => {'));
const nodes = new Map();
const stageNodes = new Map();
function node() {
  const classes = new Set();
  return { textContent: '', value: '', disabled: false, options: [], style: {}, dataset: {},
    classList: {add: (...xs) => xs.forEach(x => classes.add(x)),
      remove: (...xs) => xs.forEach(x => classes.delete(x)), contains: x => classes.has(x)},
    replaceChildren() {}, append() {}, prepend() {}, addEventListener() {}, setAttribute() {}, removeAttribute() {} };
}
let timer = 10;
const context = vm.createContext({
  console, URL, setInterval: () => ++timer, clearInterval: () => {},
  setTimeout: () => ++timer, clearTimeout: () => {},
  document: {getElementById: id => {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  }, querySelectorAll: () => [], querySelector: selector => {
    const match = selector.match(/\.step\[data-stage="([^"]+)"\]/);
    if (!match) return null;
    if (!stageNodes.has(match[1])) stageNodes.set(match[1], node());
    return stageNodes.get(match[1]);
  }, createElement: node},
});
vm.runInContext(controller, context);
const run = source => vm.runInContext(source, context);
run('loadHistory = () => {}; loadEnvironment = () => {}; loadProjects = () => {}; loadJevKey = () => {};');
async function main() {
  context.fetch = async () => { throw new Error('offline'); };
  run('state.jobId = "run-1"; state.timer = 7;');
  await run('poll()');
  assert.equal(run('state.timer'), 7, 'Network failure must keep polling alive');
  assert.match(nodes.get('runStatus').textContent, /Повторяю/);

  context.fetch = async () => ({ok: true, json: async () => ({events: [], status: 'awaiting_confirmation',
    can_confirm: true, can_discard: true, can_bridge: true, result: {}})});
  await run('poll()');
  assert.equal(run('state.timer'), 7, 'Other tabs must keep seeing merge decisions');
  assert.equal(nodes.get('discardWorktreeButton').disabled, false);

  context.fetch = async () => ({status: 404, ok: false});
  await run('poll()');
  assert.equal(run('state.jobId'), null);
  assert.equal(nodes.get('startButton').disabled, false);
  assert.equal(nodes.get('confirmMergeButton').style.display, 'none');

  const saves = [];
  let release;
  context.fetch = async (_url, options) => {
    const saved = JSON.parse(options.body).settings;
    saves.push(saved);
    if (saves.length === 1) await new Promise(resolve => { release = resolve; });
    return {ok: true, json: async () => ({settings: saved})};
  };
  run('state.status = {}; state.settingsLoaded = true; state.version = 1; collectSettings = () => ({version: state.version});');
  const first = run('saveSettings(false)');
  run('state.version = 2;');
  await run('saveSettings(false)');
  release();
  await first;
  assert.deepEqual(saves, [{version: 1}, {version: 2}], 'An edit during save must not get lost');

  // Ручной мост через обычный ChatGPT: панель показывает промпт и отправляет ответ.
  const manualRequest = {kind: 'code', stage: 'implementation', role: 'Разработчик',
    title: 'Правки кода для обычного ChatGPT', instructions: 'Верни файлы целиком.',
    prompt: 'Ты — инженер...', chars: 120, url: 'https://chatgpt.com/?q=abcdef',
    prompt_in_url: true, created_at: '2026-01-01T00:00:01+00:00'};
  context.fetch = async () => ({ok: true, json: async () => ({events: [], status: 'awaiting_answer',
    can_confirm: false, can_discard: false, can_bridge: false, result: {},
    manual: {waiting: true, answered: 0, timeout_seconds: 3600, request: manualRequest, history: []},
    can_answer: true})});
  run('state.jobId = "run-1"; state.lastEvent = 0; state.polling = false;');
  await run('poll()');
  assert.equal(nodes.get('relayPanel').hidden, false, 'A pending manual step must open the relay panel');
  assert.equal(nodes.get('relayPromptText').value, 'Ты — инженер...');
  assert.equal(nodes.get('relaySendButton').disabled, true, 'An empty answer cannot be sent');
  run('state.manualBusy = false; $("relayAnswer").value = "### FILE: app.py"; syncRelayButtons();');
  assert.equal(nodes.get('relaySendButton').disabled, false, 'A pasted answer enables sending');
  let answerBody = null;
  context.fetch = async (url, options) => {
    answerBody = {url, body: JSON.parse(options.body)};
    return {ok: true, json: async () => ({run: 'run-1', accepted: true, chars: 19})};
  };
  await run('sendRelayAnswer()');
  assert.match(answerBody.url, /\/api\/runs\/run-1\/answer$/);
  assert.deepEqual(answerBody.body, {answer: '### FILE: app.py'});
  assert.equal(nodes.get('relayPanel').hidden, true, 'The panel closes after the answer is delivered');
  assert.match(html, /id="relayPanel"/);
  assert.match(html, /value="chatgpt"/);
  assert.match(html, /id="limitFallback"/);

  run('state.jobId = null; applySettings = () => {}; showRolePrompt = () => {};');
  context.fetch = async url => ({ok: true, json: async () => url === '/api/status'
    ? {active_job: 'restored-run', professions: [], settings: {}, role_prompts: {}}
    : {events: [], status: 'running', can_confirm: false, can_discard: false, can_bridge: false}});
  await run('loadStatus()');
  assert.equal(run('state.jobId'), 'restored-run', 'Reload must reconnect to the active job');
  assert.ok(run('state.timer'));

  context.chatTestEvent = {id: 77, event: 'stage.started', stage: 'planning', role: 'Аналитик',
    message: 'Формирую план', data: {}};
  run('resetChat(); addChatMessage("user", "Вы", "Сделай чат"); addChatEvent(chatTestEvent); addChatEvent(chatTestEvent);');
  assert.equal(run('state.chatMessageCount'), 2, 'A stage event must appear once in the chat');
  assert.equal(nodes.get('chatState').textContent, 'План');
  assert.equal(run('state.chatEventIds.has("77")'), true);

  run('stepIds.forEach(id => setStage(id, "")); setStage("planning", "skipped"); setStage("review", "skipped"); activateStage("testing");');
  assert.equal(stageNodes.get('planning').classList.contains('skipped'), true, 'Quick mode must show planning as skipped, not done');
  assert.equal(stageNodes.get('review').classList.contains('skipped'), true, 'Quick mode must show review as skipped, not done');
  assert.equal(stageNodes.get('implementation').classList.contains('done'), true);
  assert.match(html, /Диалог с агентом/);
  assert.match(html, /id="chatMessages" role="log"/);
  assert.match(html, /id="newChatGPTButton"/);
  assert.match(html, /codex:\/\/plugins\/install\/build-web-data-visualization\?marketplace=openai-curated/);
  assert.match(html, /codex:\/\/plugins\/install\/game-studio\?marketplace=openai-curated/);
  assert.match(html, /codex:\/\/plugins\/install\/build-web-apps\?marketplace=openai-curated/);

  let opened;
  let copied = '';
  const popup = {location: {}, close() { this.closed = true; }};
  context.window = {open: (url, target) => { opened = {url, target}; return popup; }};
  context.navigator = {clipboard: {writeText: async text => { copied = text; }}};
  context.document.getElementById('repo').value = '/workspace/project';
  context.document.getElementById('task').value = 'Add a preferences panel';
  context.document.getElementById('checks').value = 'python -m unittest discover -s tests -t .';
  context.fetch = async (_url, options) => {
    assert.equal(JSON.parse(options.body).task, 'Add a preferences panel');
    return {ok: true, json: async () => ({url: 'codex://new?path=%2Fworkspace%2Fproject&prompt=prefilled',
      prompt: 'prefilled', prompt_in_url: true})};
  };
  await run('openNewChatGPTTask()');
  assert.equal(opened.url, 'about:blank');
  assert.equal(opened.target, '_blank');
  assert.equal(popup.location.href, 'codex://new?path=%2Fworkspace%2Fproject&prompt=prefilled');
  assert.equal(copied, '', 'Short tasks should be prefilled through the supported deep link');

  const longPrompt = 'Long complete prompt';
  context.fetch = async () => ({ok: true, json: async () => ({url: 'codex://new?path=%2Fworkspace%2Fproject',
    prompt: longPrompt, prompt_in_url: false})});
  await run('openNewChatGPTTask()');
  assert.equal(copied, longPrompt, 'A long prompt should be copied intact for manual paste');
  assert.equal(popup.location.href, 'codex://new?path=%2Fworkspace%2Fproject');
  console.log('Dashboard regression checks passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
