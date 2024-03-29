# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['src/gui.py'],
             pathex=['/Users/yotam/Work/GMU/posterization/code/src'],
             binaries=[('gco-patch/libcgco.cpython-38-darwin.so','gco')],
             datas=[('src/posterization/car.jpg','posterization')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Posterization GUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          icon='assets/icon/icon.icns')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Posterization GUI')
app = BUNDLE(coll,
             name='Posterization GUI.app',
             icon='assets/icon/icon.icns',
             bundle_identifier=None)
