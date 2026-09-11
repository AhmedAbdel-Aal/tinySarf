import {chromium} from 'playwright';
import {serveBenchmark} from './server';
const server=await serveBenchmark();
const browser=await chromium.launch({executablePath:process.env.BROWSER_PATH??'/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',headless:true});
try {
 const page=await browser.newPage();page.on('console',m=>console.log('browser',m.type(),m.text()));page.on('pageerror',e=>console.log('pageerror',e.message));
 await page.goto(server.url);await page.waitForFunction(()=>!!window.lab);
 console.log('BROWSER',browser.version());console.log('ENVIRONMENT',await page.evaluate(()=>window.lab.environment()));
 const r=await page.evaluate(()=>window.lab.parity(['ك','كتب','وبكتابهم','ك'.repeat(32)],true));console.log(JSON.stringify(r,null,2));
 console.log('PUBLIC',await page.evaluate(()=>window.lab.publicRun(['وبكتابهم'])));
} finally {await browser.close();await server.close();}
