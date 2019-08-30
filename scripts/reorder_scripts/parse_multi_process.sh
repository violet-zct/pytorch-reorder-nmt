#!/usr/bin/env bash
tot=${1}
for ((i=0; i <= ${tot} ; i++))
do
   enju -xml < ./split_files/"${i}".out >  ./split_files/"${i}".xml.parse &
   done
   wait;
