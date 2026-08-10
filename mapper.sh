#!/bin/bash
while IFS=$'\t' read -r zone_id timestamp year month dow hour rush weekend pickups rest; do
  if [ "$zone_id" = "zone_id" ]; then continue; fi
  date="${timestamp:0:10}"
  if [ -n "$pickups" ] && [ "$pickups" != "0" ] && [ "$pickups" != "0.0" ]; then
    printf "%s\t%s\t%s\n" "$zone_id" "$date" "$pickups"
  fi
done
