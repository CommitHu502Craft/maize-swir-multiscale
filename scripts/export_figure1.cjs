// Render Figure 1 SVG to vector PDF and 600-DPI PNG using headless Edge/Playwright.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

const bundled = path.join(
  os.homedir(),
  '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
);
const runtime = process.env.CODEX_NODE_MODULES || (fs.existsSync(bundled) ? bundled : null);
const { chromium } = require(runtime ? path.join(runtime, 'playwright') : 'playwright');

(async () => {
  const file = process.argv[2];
  if (!file) throw new Error('Usage: node export_figure1.cjs <path/to/figure.svg>');
  const stem = file.replace(/\.svg$/i, '');
  const svg = fs.readFileSync(file, 'utf8');
  const dims = svg.match(/width="([\d.]+)mm"\s+height="([\d.]+)mm"/);
  if (!dims) throw new Error('Expected physical SVG dimensions in mm');
  const w = Number(dims[1]);
  const h = Number(dims[2]);

  const edge =
    process.env['ProgramFiles(x86)'] &&
    path.join(process.env['ProgramFiles(x86)'], 'Microsoft/Edge/Application/msedge.exe');
  const exe = process.env.V4_CHROMIUM_EXE || (edge && fs.existsSync(edge) ? edge : null);

  const browser = await chromium.launch({
    headless: true,
    ...(exe ? { executablePath: exe } : {}),
  });
  try {
    const page = await browser.newPage({
      viewport: {
        width: Math.ceil((w * 96) / 25.4),
        height: Math.ceil((h * 96) / 25.4),
      },
      deviceScaleFactor: 600 / 96,
    });
    await page.setContent(
      `<style>@page{size:${w}mm ${h}mm;margin:0}html,body{margin:0;padding:0;background:white}svg{display:block}</style>${svg}`
    );
    await page.evaluate(() => document.fonts.ready);
    await page.pdf({
      path: stem + '.pdf',
      width: w + 'mm',
      height: h + 'mm',
      printBackground: true,
      preferCSSPageSize: true,
    });
    await page.locator('svg').first().screenshot({ path: stem + '.png' });
    console.log(`Rendered ${stem}.pdf and ${stem}.png (${w}mm x ${h}mm @ 600 DPI)`);
  } finally {
    await browser.close();
  }
})().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
