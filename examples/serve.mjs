import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
const routes=new Map([['/',new URL('./index.html',import.meta.url)],['/tinysarf/index.js',new URL('../packages/core/dist/index.js',import.meta.url)],['/tinysarf/checkpoint.js',new URL('../packages/core/dist/checkpoint.js',import.meta.url)]]);
const server=createServer(async(req,res)=>{
 const url=new URL(req.url,'http://localhost'),file=routes.get(url.pathname);
 if(!file){res.writeHead(404);res.end('Not found');return;}
 try {const body=await readFile(file);res.writeHead(200,{'Content-Type':file.pathname.endsWith('.js')?'text/javascript; charset=utf-8':'text/html; charset=utf-8','Cache-Control':'no-store'});res.end(body);}
 catch{res.writeHead(500);res.end('Run pnpm build:core first.');}
});
server.listen(4173,'127.0.0.1',()=>console.log('TinySarf example: http://localhost:4173'));
process.on('SIGINT',()=>server.close());
