# VHS Demo — hackernews-simulator
# https://github.com/charmbracelet/vhs

Output demo.gif
Output demo.webm

Set Shell "zsh"
Set FontSize 14
Set Width 1000
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 40ms

# Title
Type "# hackernews-simulator — predict HN reactions before you post"
Enter
Sleep 1s

# Predict command
Type "hn-sim predict --title 'Show HN: I built an ML model that predicts HN reactions' --no-comments"
Sleep 500ms
Enter
Sleep 8s

# Clear and show with comments
Type "clear"
Enter
Sleep 500ms

Type "# Now with simulated comments..."
Enter
Sleep 1s

Type "hn-sim predict --title 'Show HN: I built an ML model that predicts HN reactions'"
Sleep 500ms
Enter
Sleep 30s

# Clear and compare
Type "clear"
Enter
Sleep 500ms

Type "# Compare multiple titles..."
Enter
Sleep 1s

Type "hn-sim predict --title 'Why Rust is the future of systems programming' --no-comments"
Sleep 500ms
Enter
Sleep 8s

Sleep 3s
