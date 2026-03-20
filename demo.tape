# VHS Demo — hackernews-simulator

Output demo.gif
Output demo.webm

Set Shell "zsh"
Set FontSize 14
Set Width 1000
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 40ms

# Activate venv first
Hide
Type "cd /Users/malakhov/code/hn-simulator && source .venv/bin/activate"
Enter
Sleep 2s
Show

# Predict without comments
Type "echo '=== hackernews-simulator — predict HN reactions before you post ==='"
Enter
Sleep 1s

Type "hn-sim predict --title 'Show HN: I built an ML model that predicts HN reactions' --no-comments"
Sleep 500ms
Enter
Sleep 12s

Sleep 2s

# Compare another title
Type "hn-sim predict --title 'Why Rust is the future of systems programming' --no-comments"
Sleep 500ms
Enter
Sleep 12s

Sleep 3s
