module.exports = {
  apps : [{
    name: "ai-toolkit",
    script: "npm.cmd",
    args: "run build_and_start",
    cwd: "./ui", 
    env: {
      BNB_CUDA_VERSION: "130",
      PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
    }
  }]
}
