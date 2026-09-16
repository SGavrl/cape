{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = with pkgs; [
    python311
    python311Packages.pip
    python311Packages.virtualenv
    git

    stdenv.cc.cc.lib
    zlib
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
    ]}:$LD_LIBRARY_PATH"
  '';
}
