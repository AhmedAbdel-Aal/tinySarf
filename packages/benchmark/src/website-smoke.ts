import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import path from 'node:path';
import {ROOT,provenance} from './artifact';
const url=process.env.TINYSARF_WEBSITE_URL??'http://127.0.0.1:3001';
const browser=await chromium.launch({executablePath:process.env.BROWSER_PATH??'/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',headless:true});
const directory=path.join(ROOT,'packages/benchmark/local');await mkdir(directory,{recursive:true});const errors:string[]=[],checks:any={};
try{
 const page=await browser.newPage({viewport:{width:1440,height:1100}});page.on('pageerror',e=>errors.push(e.message));
 const requests:string[]=[];page.on('request',r=>requests.push(r.url()));
 await page.goto(url);await page.getByRole('heading',{name:'Look inside a word.'}).waitFor();
 await page.getByRole('button',{name:'Analyze word'}).click();await page.locator('.word-result').waitFor();await page.getByRole('button',{name:'Analyze word'}).waitFor();
 checks.webgpuSingle=(await page.locator('.morpheme').count())>0;
 checks.alternatives=await page.locator('.alternative').count();await page.locator('.alternative').nth(1).click();
 await page.getByRole('tab',{name:'JSON',exact:true}).click();checks.json=(await page.locator('.json-output').textContent())?.includes('"spans"');
 await page.getByRole('tab',{name:'Visual',exact:true}).click();
 await page.screenshot({path:path.join(directory,'website-desktop.png'),fullPage:true});
 await page.getByRole('combobox',{name:'Run with'}).click();await page.getByRole('option',{name:'CPU reference'}).click();
 await page.getByRole('tab',{name:'Batch',exact:true}).click();await page.getByRole('button',{name:'Analyze batch'}).click();await page.waitForFunction(()=>document.querySelectorAll('.word-result').length===4);checks.cpuBatch=true;
 await page.getByRole('tab',{name:'Single word',exact:true}).click();await page.getByLabel('Your Arabic word').fill('invalid');await page.getByRole('button',{name:'Analyze word'}).click();await page.getByRole('alert').waitFor();checks.invalidInput=true;
 await page.getByLabel('Your Arabic word').fill('وَبِكِتَابِهِمْ');await page.getByRole('button',{name:'Analyze word'}).click();await page.locator('.word-result').waitFor();checks.diacritized=true;
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(directory,'website-mobile.png'),fullPage:true});
 checks.mobileNoHorizontalOverflow=await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth);
 checks.localRequestsOnly=requests.every(r=>new URL(r).origin===new URL(url).origin);
 for(const document of ['/docs/MODEL_CARD.md','/docs/architecture.md','/results/current.json'])checks[document]=(await page.request.get(url+document)).status()===200;
 const report={...provenance(),kind:'website-smoke',url,checks,errors,passed:!errors.length&&Object.values(checks).every(Boolean)};
 await writeFile(path.join(directory,'website-smoke.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!report.passed)throw new Error('Website smoke failed');
}finally{await browser.close();}
