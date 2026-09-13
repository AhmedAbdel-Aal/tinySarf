import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  async redirects() {
    return [{ source: '/mn', destination: '/', permanent: true }];
  },
};

export default nextConfig;
