import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  // don't let `next dev` write AGENTS.md into the repo
  agentRules: false,
};

export default config;
