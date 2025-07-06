---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.

# Error Prevention Instructions
## Powershell Execution
- When executing commands in PowerShell, ensure that the syntax is compatible with the version of PowerShell being used.
- Use the correct command separators and syntax for the version of PowerShell to avoid errors.
- If you encounter an error related to command separators, such as "The token '&&' is not a valid statement separator in this version," replace '&&' with a semicolon ';' or use separate lines for commands.
- The issue with PowerShell commands occurs because I've been using && as a command separator, which is valid in bash/Unix shells but not in Windows PowerShell. In PowerShell, && is not recognized as a valid statement separator. 
- Instead, PowerShell uses ; (semicolon) to chain commands on the same line, or you can use separate commands on different lines.
- When executing powershell scripts, be sure to use the correct syntax for this version of powershell.
Example:
---bash
PS S:\Maker Stuff\Programming\V7P3R Chess Engine\viper_chess_engine> cd "S:\Maker Stuff\Programming\V7P3R Chess Engine\viper_chess_engine\metrics" && python chess_metrics.py
At line:1 char:79
+ ... Programming\V7P3R Chess Engine\viper_chess_engine\metrics" && python  ...
+                                                                ~~
The token '&&' is not a valid statement separator in this version.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : InvalidEndOfLine
---