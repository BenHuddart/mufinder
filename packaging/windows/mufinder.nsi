; MuFinder Windows Installer (NSIS)
;
; Build variables (passed on the makensis command line):
;   VERSION     — M.N.P version string (e.g. 0.1.2 for releases, 0.0.<run> for previews)
;   APP_DIR     — absolute path to the PyInstaller onedir output (dist\MuFinder)
;   ICON_FILE   — (optional) absolute path to .ico; defaults to packaging\build\mufinder.ico
;   OUTPUT_FILE — absolute output installer path

!define APP_NAME "MuFinder"
!define EXECUTABLE_NAME "MuFinder.exe"
!define INSTALLED_ICON_NAME "mufinder.ico"
!define PUBLISHER "MuFinder Contributors"

!ifdef ICON_FILE
  !define APP_ICON "${ICON_FILE}"
!else
  !define APP_ICON "packaging\build\mufinder.ico"
!endif

!include "MUI2.nsh"
!include "WordFunc.nsh"
!insertmacro VersionCompare

Name "${APP_NAME} ${VERSION}"
OutFile "${OUTPUT_FILE}"
InstallDir "$LOCALAPPDATA\Programs\${APP_NAME}"
InstallDirRegKey HKCU "Software\${APP_NAME}" "InstallDir"
RequestExecutionLevel user

Icon "${APP_ICON}"
!define MUI_ICON "${APP_ICON}"
!define MUI_UNICON "${APP_ICON}"

!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

; Enumerate HKCU Uninstall keys and msiexec /x any GUID-keyed entry whose
; DisplayName matches ${APP_NAME}.  Handles users upgrading from the WiX MSI
; release, which registered under a GUID-based key that NSIS does not know about.
Function RemoveOrphanedMsiEntry
  StrCpy $0 0
  loop:
    EnumRegKey $1 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall" $0
    StrCmp $1 "" done
    StrCpy $2 $1 1
    StrCmp $2 "{" 0 next
    ReadRegStr $3 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\$1" "DisplayName"
    StrCmp $3 "${APP_NAME}" 0 next
    ExecWait '"msiexec.exe" /x "$1" /quiet /norestart'
    StrCpy $0 0
    Goto loop
  next:
    IntOp $0 $0 + 1
    Goto loop
  done:
FunctionEnd

; Abort if the currently installed NSIS version is newer than ${VERSION}.
Function CheckDowngrade
  ReadRegStr $0 HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion"
  StrCmp $0 "" done
  ${VersionCompare} "$0" "${VERSION}" $1
  StrCmp $1 "1" 0 done
  MessageBox MB_OK|MB_ICONSTOP "A newer version of ${APP_NAME} ($0) is already installed.$\nSetup will now exit."
  Quit
  done:
FunctionEnd

Section "Main" SEC_MAIN
  Call RemoveOrphanedMsiEntry
  Call CheckDowngrade

  ; Close any running instance so the directory wipe does not leave stale locked files
  ExecWait '"taskkill" /im "${EXECUTABLE_NAME}"'
  Sleep 1000

  RMDir /r "$INSTDIR"
  SetOutPath "$INSTDIR"

  File /r "${APP_DIR}\*"
  File /oname="${INSTALLED_ICON_NAME}" "${APP_ICON}"

  CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${EXECUTABLE_NAME}" "" "$INSTDIR\${INSTALLED_ICON_NAME}" 0
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${EXECUTABLE_NAME}" "" "$INSTDIR\${INSTALLED_ICON_NAME}" 0

  WriteRegStr HKCU "Software\${APP_NAME}" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\${APP_NAME}" "installed" "1"

  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName" "${APP_NAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "Publisher" "${PUBLISHER}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayIcon" "$INSTDIR\${INSTALLED_ICON_NAME}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "InstallLocation" "$INSTDIR"
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "NoRepair" 1

  WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Uninstall"
  RMDir /r "$INSTDIR"

  Delete "$DESKTOP\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  RMDir "$SMPROGRAMS\${APP_NAME}"

  DeleteRegKey HKCU "Software\${APP_NAME}"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
SectionEnd
