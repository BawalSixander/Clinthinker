#ifndef AppVersion
#define AppVersion "0.0.0"
#endif

[Setup]
AppId={{1C9311D5-89C5-4A9F-A62C-CC47E147F52A}
AppName=Clinthinker
AppVersion={#AppVersion}
AppPublisher=seanlordtrillo-ops
DefaultDirName={localappdata}\Programs\Clinthinker
DefaultGroupName=Clinthinker
PrivilegesRequired=lowest
OutputDir=release
OutputBaseFilename=Clinthinker-Setup-{#AppVersion}
SetupIconFile=assets\app-icon.ico
UninstallDisplayIcon={app}\Clinthinker©.exe
CloseApplications=yes
CloseApplicationsFilter=Clinthinker©.exe
RestartApplications=no
ArchitecturesInstallIn64BitMode=x64os
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "release\Clinthinker©\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Clinthinker"; Filename: "{app}\Clinthinker©.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Clinthinker"; Filename: "{app}\Clinthinker©.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Clinthinker©.exe"; Description: "Launch Clinthinker"; Flags: postinstall nowait skipifsilent
