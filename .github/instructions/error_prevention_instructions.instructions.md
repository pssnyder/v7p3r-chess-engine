---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.

# Error Prevention Instructions
## EXTREMELY IMPORTANT - MUST FOLLOW
- YOU WILL NOT use "&&" and instead use proper PowerShell syntax.

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

### Additional Example
#### DO NOT MAKE THIS MISTAKE
```powershell
PS S:\Maker Stuff\Programming\V7P3R Chess Engine\v7p3r_chess_engine> cd "s:\Maker Stuff\Programming\V7P3R Chess Engine\v7p3r_chess_engine" && "C:/Users/patss/AppData/Local/Programs/Python/Python312/python.exe" testing/test_phase5_final_polish.py
At line:1 char:71
+ ... r Stuff\Programming\V7P3R Chess Engine\v7p3r_chess_engine" && "C:/Use ...
+                                                                ~~
The token '&&' is not a valid statement separator in this version.
At line:1 char:74
+ ... _engine" && "C:/Users/patss/AppData/Local/Programs/Python/Python312/p ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
Expressions are only allowed as the first element of a pipeline.
At line:1 char:142
+ ... rams/Python/Python312/python.exe" testing/test_phase5_final_polish.py    
+                                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unexpected token 'testing/test_phase5_final_polish.py' in expression or statement.
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : InvalidEndOfLine
```