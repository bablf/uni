#!/bin/bash
git pull
git add *
git status
read -p "Are you sure? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    git commit -m "aktuelle updates"
    git push
    git status
fi
