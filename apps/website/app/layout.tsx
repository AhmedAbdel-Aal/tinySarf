import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = {icons:{icon:'/favicon.svg'},title:'TinySarf — Arabic morphology lab',description:'Inspect Arabic morphemes, roots and patterns locally in your browser. An experimental learned WebGPU model with reproducible evidence.'};
export default function RootLayout({children}:{children:React.ReactNode}) {return <html lang="en"><body>{children}</body></html>;}
