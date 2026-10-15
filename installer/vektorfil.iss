; Inno Setup script for VektorFIL -- compiled by tools/build_windows.py, which passes
;   /DMyAppVersion=<version> /DSourceDir=<Nuitka dist folder> /DOutputDir=<build folder>
;   /DRootDir=<project root>
; Keep AppId unchanged between releases so upgrades replace the previous version.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\build\VektorFIL.dist"
#endif
#ifndef OutputDir
  #define OutputDir "..\build"
#endif
#ifndef RootDir
  #define RootDir ".."
#endif

#define MyAppName "VektorFIL"
#define MyAppPublisher "Pieter Ibelings"
#define MyAppURL "https://github.com/ibelinp/vektorfil"
#define MyAppExeName "VektorFIL.exe"

[Setup]
AppId={{1C09FFFD-C3B3-4CE8-AD6A-4BC3036A9722}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
AppCopyright=Copyright (C) 2026 Pieter Ibelings
VersionInfoVersion={#MyAppVersion}
VersionInfoDescription={#MyAppName} setup
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; per-user install without administrator rights by default; all-users install on request
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile={#RootDir}\LICENSE
SetupIconFile={#RootDir}\assets\vektorfil.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
OutputDir={#OutputDir}
OutputBaseFilename={#MyAppName}-{#MyAppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
; PyInstaller keeps bundled data in the _internal subfolder
Name: "{group}\{#MyAppName} User Guide"; Filename: "{app}\_internal\docs\index.html"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
