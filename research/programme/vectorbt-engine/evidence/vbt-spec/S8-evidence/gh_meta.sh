#!/bin/bash
# usage: gh_meta.sh owner/repo ... ; prints TSV: repo stars pushed_at license archived open_issues latest_release(date) desc
for r in "$@"; do
  j=$(gh api repos/$r 2>/dev/null) || { echo -e "$r\tNOTFOUND"; continue; }
  rel=$(gh api repos/$r/releases/latest --jq ".tag_name+\" \"+.published_at[:10]" 2>/dev/null | grep -v message || echo "norelease")
  lc=$(gh api "repos/$r/commits?per_page=1" --jq '.[0].commit.committer.date[:10]' 2>/dev/null || echo "?")
  echo "$j" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('\t'.join(str(x) for x in ['$r',d['stargazers_count'],'lastcommit=$lc',(d.get('license') or {}).get('spdx_id'),'archived='+str(d['archived']),'issues='+str(d['open_issues_count']),'rel=$rel',(d.get('description') or '')[:70]]))"
done
