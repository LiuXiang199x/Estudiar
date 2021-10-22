#!/usr/bin/env bash
#coding:utf-8


# alias命令用来为一个命令指定别名，这样更便于记忆。
alias echo='echo it says: '


# 命令执行结果

if cd $home; then
  rm 11.py
# cd $home && rm 11.py

else
  echo "Could not change directory! Aborting." 1>&2
  exit 1
fi
# cd $some_directory || exit 1


# exit 1

