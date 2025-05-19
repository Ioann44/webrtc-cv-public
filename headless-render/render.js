const puppeteer = require('puppeteer');

const width = parseInt(process.argv[2]) || 512;
const height = parseInt(process.argv[3]) || 512;
const data = process.argv[4] || "";

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox'] });
  const page = (await browser.newPage()).on('console', message => console.error(`${message.type().substr(0, 3).toUpperCase()} ${message.text()}`));
  await page.setViewport({ width: width, height: height });
  await page.exposeFunction("getCustomData", () => data);
  await page.goto('file://' + process.cwd() + '/scene.html');

  await page.evaluate(async () => {
    if (typeof window.renderScene === 'function') {
      await window.renderScene();
    }
  });

  const buffer = await page.screenshot({ encoding: 'base64' });
  await browser.close();
  console.log(buffer);
})();
