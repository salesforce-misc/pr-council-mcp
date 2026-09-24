from pr_council.review.diff import commentable_lines


def test_commentable_lines_contains_only_added_right_side_lines():
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -2,2 +2,3 @@
 context
-old
+new
+added
"""
    assert commentable_lines(diff) == {"a.py": {3, 4}}


def test_commentable_lines_ignores_deletions_and_tracks_multiple_hunks():
    diff = """diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1,3 +1,4 @@
 keep
+first
 keep2
-removed
@@ -10,1 +11,2 @@
 anchor
+second
"""
    assert commentable_lines(diff) == {"b.py": {2, 12}}
