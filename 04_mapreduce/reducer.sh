#!/bin/bash
current_key=""
current_sum=0
while IFS=$'\t' read -r zone_id date count; do
  key="${zone_id}	${date}"
  int_count="${count%.*}"
  if [ -z "$int_count" ] || [ "$int_count" = "0" ]; then continue; fi
  if [ "$key" = "$current_key" ]; then
    current_sum=$((current_sum + int_count))
  else
    if [ -n "$current_key" ]; then
      printf "%s\t%s\n" "$current_key" "$current_sum"
    fi
    current_key="$key"
    current_sum=$int_count
  fi
done
if [ -n "$current_key" ]; then
  printf "%s\t%s\n" "$current_key" "$current_sum"
fi
