// Behavioral tests of the dashboard controller in a minimal DOM (no external packages).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('ai_orchestrate/static/index.html', 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
new vm.Script(script); // Parse all code, including handlers below the controller.
const controller = script.slice(0, script.indexOf('    document.querySelectorAll(".mode-option").forEach(node => {'));
const nodes = new Map();
function node() {
  const classes = new Set();
  return { textContent: '', value: '', disabled: false, options: [], style: {}, dataset: {},
    classList: {add: (...xs) => xs.forEach(x => classes.add(x)),
      remove: (...xs) => xs.forEach(x => classes.delete(x)), contains: x => classes.has(x)},
    replaceChildren() {}, append() {}, prepend() {}, addEventListener() {} };
}
let timer = 10;
const context = vm.createContext({
  console, URL, setInterval: () => ++timer, clearInterval: () => {},
  setTimeout: () => ++timer, clearTimeout: () => {},
  document: {getElementById: id => {
    if (!nodes.has(id)) nodes.set(id, node());
    return nodes.get(id);
  }, querySelectorAll: () => [], querySelector: () => null, createElement: node},
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

  run('state.jobId = null; applySettings = () => {}; showRolePrompt = () => {};');
  context.fetch = async url => ({ok: true, json: async () => url === '/api/status'
    ? {active_job: 'restored-run', professions: [], settings: {}, role_prompts: {}}
    : {events: [], status: 'running', can_confirm: false, can_discard: false, can_bridge: false}});
  await run('loadStatus()');
  assert.equal(run('state.jobId'), 'restored-run', 'Reload must reconnect to the active job');
  assert.ok(run('state.timer'));
  console.log('Dashboard regression checks passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
