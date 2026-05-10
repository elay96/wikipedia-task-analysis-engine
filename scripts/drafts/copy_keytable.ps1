$ErrorActionPreference = 'Stop'
$src = 'G:\My Drive\4. Education\אקדמיה\הדרייב של עילי\תואר שני\תיזה\ניסויים\Wikipedia Task\הרצות\4. הרצה שלישית - 138 אנשים\KeyTable.csv'
$dst = 'C:\Users\elay9\wikipedia-task-analysis-engine\data\Spatial Search Data\KeyTable.csv'
Copy-Item -LiteralPath $src -Destination $dst -Force
Get-Item -LiteralPath $dst | Format-Table FullName, Length, LastWriteTime -AutoSize
