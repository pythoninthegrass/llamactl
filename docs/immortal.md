# Immortal process supervision

[immortal](https://immortal.run/) supervises llama-server as a daemon:
auto-restart on crash, log rotation, and persistence across reboots.

`lctl` wraps the common immortalctl commands. This document covers the
underlying setup.

## Install

```bash
brew install immortal
```

## Config directories

```bash
sudo mkdir -p /usr/local/etc/immortal /usr/local/var/log
sudo chown $(whoami):staff /usr/local/var/log
ln -s /usr/local/etc/immortal ~/.config/immortal
```

`immortaldir` watches `/usr/local/etc/immortal/` for `*.yml` files.
The symlink at `~/.config/immortal` is for convenience. The log
directory must be writable by the service user for the `user:` directive
to work (immortal creates log files after dropping privileges).

## Service config

`/usr/local/etc/immortal/llama-server.yml`:

```yaml
cmd: /Users/lance/git/llama-cpp-turboquant/build/bin/llama-server --host 0.0.0.0 --port 8080 --models-preset /Users/lance/models/models.ini
cwd: /Users/lance/git/llama-cpp-turboquant
user: lance
env:
    HOME: /Users/lance
log:
    file: /usr/local/var/log/llama-server.log
    age: 86400
    num: 7
    size: 10
    timestamp: true
stderr:
    file: /usr/local/var/log/llama-server-err.log
    age: 86400
    num: 7
    size: 10
wait: 1
```

`user: lance` tells immortal to drop privileges from root. Required for
Metal GPU access. The `stderr` section captures error output separately.

## LaunchDaemon

`/Library/LaunchDaemons/com.immortal.dir.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.immortal.dir</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/homebrew/bin/immortaldir</string>
        <string>/usr/local/etc/immortal</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/usr/local/var/log/immortaldir.log</string>
    <key>StandardErrorPath</key>
    <string>/usr/local/var/log/immortaldir.log</string>
</dict>
</plist>
```

Uses the full Homebrew path since LaunchDaemons don't inherit the user
`$PATH`. The `EnvironmentVariables/PATH` is required so `immortaldir`
can find the `immortal` binary when spawning services. Without it,
services crash-loop silently. `KeepAlive` restarts `immortaldir` itself
if it exits.

## Load and start

```bash
sudo launchctl load -w /Library/LaunchDaemons/com.immortal.dir.plist
```

## Verify

```bash
sudo immortalctl                               # list supervised services (sudo required, socket is root-owned)
curl -s http://127.0.0.1:8080/v1/models | jq . # confirm llama-server responds
```

## Direct immortalctl commands

```bash
sudo immortalctl status llama-server   # check status
sudo immortalctl stop llama-server     # stop (will not auto-restart)
sudo immortalctl start llama-server    # start again
sudo immortalctl restart llama-server  # restart
sudo immortalctl halt llama-server     # stop and prevent restart until config is touched
```

To update the service config, edit the yml and `immortaldir` picks up
the change automatically (restarts the service).

## Router mode and restarts

In router mode, `immortalctl restart` only cycles the router process.
Child model server processes survive and can cause GPU OOM if the new
router loads additional models. `lctl restart` handles this by killing
all llama-server processes (`pkill -f llama-server`) and letting
immortal auto-recover the router.

## If Metal GPU fails to initialize

A LaunchDaemon runs outside the user's GUI session. If llama-server
falls back to CPU-only (check logs for `Metal` or `ggml_metal_init`),
add `SessionCreate` to the plist:

```xml
<key>SessionCreate</key>
<true/>
```

Then reload:

```bash
sudo launchctl unload /Library/LaunchDaemons/com.immortal.dir.plist
sudo launchctl load -w /Library/LaunchDaemons/com.immortal.dir.plist
```
