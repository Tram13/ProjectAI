#!/bin/bash

cd ../data

for directory in *; do
  cd $directory;
  find -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"; rm  "$1"' _ {} \;
  find ./ -empty -type f -delete
  cd ../
done