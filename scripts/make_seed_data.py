"""Generates data/raw/nimbus_tickets.jsonl, the fictional Nimbus support dataset.

Deterministic (seeded), so re-running produces byte-identical data. The
output is checked into the repo; this script exists so the dataset is
reproducible and auditable, not magic.

Intentional dirt is baked in on purpose: exact duplicate questions, a few
one-word answers, and some malformed lines. Stages 1-2 exist to clean
exactly this kind of thing, and clean toy data would make those stages
look like theater.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "nimbus_tickets.jsonl"

# (topic tag, question template, answer): Nimbus's invented feature set.
# Facts stay consistent across rows: 30-second sync, 30-day trash, Pro at
# 4.99, three export formats, etc. Consistency matters more than volume for
# a fine-tune this small.
BASE = [
    ("sync", "How often does Nimbus sync my notes?",
     "Nimbus syncs automatically every 30 seconds while you're online. You can force a sync anytime by pulling down on the note list."),
    ("sync", "My notes aren't syncing between my phone and laptop, what do I do?",
     "First check that both devices are signed into the same Nimbus account under Settings > Account. Then pull down on the note list to force a sync. If a note still won't sync, it usually has an attachment over the 25 MB limit."),
    ("sync", "Does Nimbus work offline?",
     "Yes. You can read and edit all your notes offline. Changes are stored on the device and merged automatically the next time you're online. If the same note was edited in two places, Nimbus keeps both versions and marks the note with a conflict badge."),
    ("sync", "What happens if I edit the same note on two devices?",
     "Nimbus keeps both versions and shows a conflict badge on the note. Open it and tap the badge to compare the two versions side by side and choose which one to keep, or merge them by hand."),
    ("pin", "How do I pin a note to the top?",
     "Long-press the note in the list and choose Pin to top. Pinned notes stay above everything else and show a small pin icon. You can pin up to 10 notes per notebook."),
    ("pin", "Is there a limit to pinned notes?",
     "Yes, you can pin up to 10 notes per notebook. If you hit the limit, unpin something first by long-pressing a pinned note and choosing Unpin."),
    ("share", "How do I share a notebook with someone?",
     "Open the notebook, tap Share in the top bar, and enter their email. You can give view-only or can-edit access. They'll get an invite link that works even if they don't have a Nimbus account yet."),
    ("share", "Can I make a note public?",
     "Yes. Open the note, tap Share, and switch on Public link. Anyone with the link can view (not edit) the note. Turn the link off anytime and it stops working immediately."),
    ("share", "How do I stop sharing a notebook?",
     "Open the notebook, tap Share, and either remove individual people from the list or tap Stop sharing to revoke everyone's access at once, including public links."),
    ("trash", "How long do deleted notes stay in the trash?",
     "Deleted notes sit in Trash for 30 days before Nimbus removes them permanently. You can restore anything from Trash during those 30 days, or empty the Trash manually to delete things right away."),
    ("trash", "I deleted a note by accident, can I get it back?",
     "Yes, if it's within 30 days. Open Trash from the sidebar, find the note, and tap Restore. It goes back to the notebook it came from. After 30 days in Trash, notes are gone for good."),
    ("export", "How do I export a note?",
     "Open the note, tap the ... menu, and choose Export. Nimbus can export as Markdown, PDF, or plain text. Exporting a whole notebook produces a ZIP with one file per note."),
    ("export", "Can I export all my notes at once?",
     "Yes. Go to Settings > Data > Export everything. You'll get a ZIP containing every notebook as a folder, with each note as a Markdown file and attachments alongside."),
    ("export", "What formats can Nimbus export to?",
     "Three formats: Markdown, PDF, and plain text. Markdown keeps all formatting and is the best choice if you might import into another app later."),
    ("tags", "How do tags work in Nimbus?",
     "Type # anywhere in a note to create a tag, like #recipes. Tags appear in the sidebar under Tags, and tapping one shows every note that uses it. A note can have any number of tags."),
    ("tags", "Can I rename a tag everywhere at once?",
     "Yes. In the sidebar, long-press the tag under Tags and choose Rename. Every note using that tag updates automatically."),
    ("search", "How do I search inside my notes?",
     "Tap the magnifying glass at the top of the note list. Search covers note titles, body text, and text inside attachments like PDFs. Use quotes for exact phrases and tag: to filter by tag, like tag:work."),
    ("search", "Does search look inside PDFs?",
     "Yes. Nimbus indexes the text of PDF and image attachments (using on-device OCR for images), so searching finds matches inside them too. Indexing new attachments can take a minute."),
    ("shortcut", "What keyboard shortcuts does Nimbus have?",
     "The big ones: Ctrl+N for a new note, Ctrl+F to search, Ctrl+P to pin, Ctrl+E to export, and Ctrl+/ to see the full shortcut list. On Mac, use Cmd instead of Ctrl."),
    ("billing", "How much does Nimbus Pro cost?",
     "Nimbus Pro is 4.99 per month or 49 per year. Pro removes the 25 MB attachment limit, adds version history for 90 days, and unlocks offline notebooks on unlimited devices."),
    ("billing", "What do I get with Nimbus Pro?",
     "Pro raises the attachment limit from 25 MB to 1 GB per file, keeps 90 days of version history for every note, allows unlimited offline devices, and adds priority support."),
    ("billing", "How do I cancel my Pro subscription?",
     "Go to Settings > Account > Subscription and tap Cancel. You keep Pro features until the end of the paid period, then your account drops back to the free plan. Your notes are never deleted when you downgrade."),
    ("attach", "What's the attachment size limit?",
     "Free accounts can attach files up to 25 MB each. Nimbus Pro raises that to 1 GB per file. There's no limit on the number of attachments per note."),
    ("attach", "What file types can I attach to a note?",
     "Anything. Images and PDFs get inline previews and are searchable; other file types show as a download chip inside the note."),
    ("history", "Does Nimbus keep old versions of my notes?",
     "Free accounts keep 7 days of version history; Pro keeps 90 days. Open a note, tap the ... menu, and choose Version history to browse and restore any older version."),
    ("history", "How do I restore an older version of a note?",
     "Open the note, tap the ... menu, choose Version history, pick the version you want, and tap Restore. The current version is saved into history first, so restoring never loses anything."),
    ("template", "Can I create note templates?",
     "Yes. Write the note you want to reuse, tap the ... menu, and choose Save as template. New notes can then start from any template via New note > From template."),
    ("reminder", "Can Nimbus remind me about a note?",
     "Yes. Open the note, tap the bell icon, and pick a date and time. You'll get a notification on every signed-in device. Recurring reminders (daily, weekly, monthly) are supported too."),
    ("widget", "Is there a home screen widget?",
     "Yes, on both iOS and Android. Long-press your home screen, add the Nimbus widget, and choose between Quick note (one tap to start writing) or a pinned-notes list."),
    ("security", "Are my notes encrypted?",
     "Notes are encrypted in transit (TLS) and at rest on Nimbus servers. You can additionally lock individual notebooks with a passcode or biometrics under the notebook's Share & security settings."),
    ("security", "Can I lock a notebook with a password?",
     "Yes. Open the notebook, go to Share & security, and turn on Notebook lock. You can use a passcode or fingerprint/face unlock. Locked notebooks are excluded from search previews."),
    ("account", "How do I change my email address?",
     "Go to Settings > Account > Email, enter the new address, and confirm via the link Nimbus sends there. Your notes and sharing settings are unaffected."),
    ("account", "How do I delete my Nimbus account?",
     "Settings > Account > Delete account. Nimbus asks you to export your data first, then deletes everything after a 14-day grace period during which you can change your mind by signing back in."),
    ("import", "Can I import my notes from another app?",
     "Yes. Settings > Data > Import accepts Markdown files, plain text, and Evernote .enex exports. Folder structure becomes notebooks, and #hashtags in the text become Nimbus tags."),
    ("markdown", "Does Nimbus support Markdown?",
     "Yes. Type Markdown and it renders live: # headings, **bold**, - lists, [links](url), and ``` code blocks. There's also a formatting toolbar if you'd rather tap than type."),
]

PARAPHRASES = [
    ("sync", "Why do my notes take a while to show up on my other device?",
     "Nimbus syncs every 30 seconds while you're online, so short delays are normal. Pull down on the note list to force an immediate sync. If it never arrives, check both devices are on the same account and that the note's attachments are under your plan's size limit."),
    ("trash", "Where do notes go when I delete them?",
     "Into Trash, which you'll find in the sidebar. They stay there for 30 days and can be restored anytime in that window. After 30 days they're permanently removed."),
    ("share", "My friend can't edit the notebook I shared, why?",
     "You probably shared it as view-only. Open the notebook, tap Share, find your friend in the list, and switch their access from Can view to Can edit."),
    ("export", "I want a PDF of one of my notes.",
     "Open the note, tap the ... menu, choose Export, and pick PDF. The PDF keeps your formatting, images, and checklists exactly as they appear in the note."),
    ("billing", "Is there a free trial for Pro?",
     "Yes, 14 days, no card required. Start it from Settings > Account > Try Pro. When the trial ends you drop back to the free plan automatically unless you subscribe, nothing is charged silently."),
    ("search", "Can I search only within one notebook?",
     "Yes. Open that notebook first and then tap the magnifying glass. Search is scoped to the open notebook. From the All notes view, search covers everything instead."),
    ("reminder", "How do I set a repeating reminder on a note?",
     "Open the note, tap the bell icon, pick the first date and time, then tap Repeat and choose daily, weekly, or monthly. Each occurrence notifies every device you're signed into."),
    ("security", "Does Nimbus staff read my notes?",
     "No. Notes are encrypted at rest and access is technically restricted; support staff can only see note metadata (like counts and sizes) when you open a support ticket, never note contents."),
]


# More ways real users ask about the same features. Same facts, new wording,
# which is what teaches a small model the domain rather than a few strings.
# Also big enough that the 10% validation split has a meaningful number of rows.
MORE = [
    # sync
    ("sync", "How do I force Nimbus to sync right now?",
     "Pull down on the note list. That triggers an immediate sync instead of waiting for the automatic one, which runs every 30 seconds while you're online."),
    ("sync", "Is there a sync button?",
     "There's no separate button: pull down on the note list to force a sync. Otherwise Nimbus syncs on its own every 30 seconds while you're online."),
    ("sync", "One note won't sync but the rest do. Why?",
     "That usually means the note has an attachment over your plan's size limit, 25 MB on the free plan or 1 GB on Pro. Remove or shrink the attachment and pull down on the note list to sync again."),
    ("sync", "I see a conflict badge on a note. What does it mean?",
     "The note was edited on two devices before they synced. Nimbus kept both versions. Tap the badge to compare them side by side, then keep one or merge them by hand."),
    ("sync", "Will I lose edits I make on a plane?",
     "No. Offline edits are stored on the device and merged automatically the next time you're online. If the same note changed elsewhere meanwhile, Nimbus keeps both versions and shows a conflict badge."),
    ("sync", "My phone and laptop show different notes.",
     "Check that both devices are signed into the same Nimbus account under Settings > Account, then pull down on the note list on each to force a sync."),
    ("sync", "Does Nimbus sync over mobile data?",
     "Yes. Nimbus syncs every 30 seconds whenever you're online, on Wi-Fi or mobile data. You can always force a sync by pulling down on the note list."),
    # pin
    ("pin", "How do I unpin a note?",
     "Long-press the pinned note in the list and choose Unpin. It drops back into its normal place in the notebook."),
    ("pin", "Why can't I pin another note?",
     "You've probably reached the limit of 10 pinned notes per notebook. Unpin one first by long-pressing it and choosing Unpin, then pin the new note."),
    ("pin", "Is there a shortcut to pin a note on desktop?",
     "Yes, Ctrl+P pins the open note, or Cmd+P on Mac. On mobile, long-press the note in the list and choose Pin to top."),
    ("pin", "How can I tell which notes are pinned?",
     "Pinned notes sit above everything else in the note list and show a small pin icon. Each notebook can have up to 10 of them."),
    ("pin", "Can I keep my most important notes at the top of the list?",
     "Yes, pin them. Long-press a note and choose Pin to top. Pinned notes always stay above the rest, up to 10 per notebook."),
    # share
    ("share", "Can someone without a Nimbus account see my shared notebook?",
     "Yes. When you share a notebook by email, the invite link works even if they don't have a Nimbus account yet."),
    ("share", "What's the difference between view-only and can-edit?",
     "View-only people can read the notebook but not change it. Can-edit people can add, edit and delete notes in it. Change anyone's access from the notebook's Share screen."),
    ("share", "How do I turn off a public link?",
     "Open the note, tap Share, and switch off Public link. The link stops working immediately for everyone who has it."),
    ("share", "Can people edit a note through its public link?",
     "No. Public links are view-only. To let someone edit, share the notebook with them by email and give them can-edit access."),
    ("share", "How do I remove one person from a shared notebook?",
     "Open the notebook, tap Share, and remove that person from the list. Everyone else keeps their access."),
    ("share", "I want to revoke everyone's access to a notebook.",
     "Open the notebook, tap Share, and tap Stop sharing. That removes every person and turns off any public links at once."),
    ("share", "How do I give my coworker edit access?",
     "Open the notebook, tap Share in the top bar, enter their email, and choose can-edit. If they're already on the list, switch their access from Can view to Can edit."),
    # trash
    ("trash", "Where is the Trash?",
     "Trash is in the sidebar. Deleted notes stay there for 30 days, and you can restore any of them during that time."),
    ("trash", "How do I permanently delete a note right away?",
     "Delete the note, then open Trash from the sidebar and empty it. Otherwise Nimbus removes notes from Trash automatically after 30 days."),
    ("trash", "I deleted a note two months ago. Can I recover it?",
     "Unfortunately not. Notes stay in Trash for 30 days and are then removed permanently, so a note deleted two months ago is gone for good."),
    ("trash", "Where does a restored note go?",
     "Back to the notebook it came from. Open Trash from the sidebar, find the note, and tap Restore."),
    ("trash", "Does emptying the trash free up space?",
     "Yes. Emptying Trash deletes those notes and their attachments right away instead of waiting for the automatic 30-day cleanup."),
    ("trash", "Does Nimbus delete old notes on its own?",
     "Only notes you've already deleted. They sit in Trash for 30 days and are then removed permanently. Notes you haven't deleted are never removed automatically."),
    # export
    ("export", "Can I export a notebook as a ZIP?",
     "Yes. Exporting a whole notebook produces a ZIP with one file per note, in Markdown, PDF or plain text."),
    ("export", "Can I back up everything I have in Nimbus?",
     "Yes. Go to Settings > Data > Export everything. You get a ZIP with every notebook as a folder, each note as a Markdown file, and attachments alongside."),
    ("export", "Which export format should I pick if I'm moving to another app?",
     "Markdown. It keeps all your formatting and most note apps can import it."),
    ("export", "Can I export a note as a Word document?",
     "Not directly. Nimbus exports to Markdown, PDF and plain text. Markdown keeps your formatting best if you plan to convert it later."),
    ("export", "Is there a keyboard shortcut for export?",
     "Yes, Ctrl+E exports the open note, or Cmd+E on Mac. You can also use the ... menu and choose Export."),
    ("export", "Do exports include attachments?",
     "Export everything, under Settings > Data, puts attachments in the ZIP alongside the Markdown files for each note."),
    # tags
    ("tags", "How do I add a tag to a note?",
     "Type # followed by the tag name anywhere in the note, like #work. The tag appears in the sidebar under Tags right away."),
    ("tags", "How do I see all notes with a certain tag?",
     "Open the sidebar, look under Tags, and tap the tag. Nimbus shows every note that uses it."),
    ("tags", "Is there a limit on tags per note?",
     "No. A note can have any number of tags. Just type # and the tag name wherever you like in the note."),
    ("tags", "I misspelled a tag on dozens of notes. How do I fix it?",
     "Long-press the tag under Tags in the sidebar and choose Rename. Every note using it updates automatically."),
    ("tags", "Do hashtags from imported notes become tags?",
     "Yes. When you import through Settings > Data > Import, #hashtags in the text become Nimbus tags."),
    # search
    ("search", "How do I search for an exact phrase?",
     "Put the phrase in quotes in the search box, like \"quarterly plan\". Tap the magnifying glass at the top of the note list to start searching."),
    ("search", "How do I search only notes with a specific tag?",
     "Add tag: to your search, like tag:work budget. That limits results to notes with that tag."),
    ("search", "Can Nimbus find text inside a photo?",
     "Yes. Nimbus runs on-device OCR on image attachments, so search finds text inside photos and scans. New attachments can take a minute to index."),
    ("search", "I just attached a PDF but search doesn't find it.",
     "Give it a minute. Nimbus indexes the text of new PDF and image attachments in the background, and new files can take a moment before they show up in search."),
    ("search", "What's the shortcut to search?",
     "Ctrl+F on Windows and Linux, Cmd+F on Mac. On mobile, tap the magnifying glass at the top of the note list."),
    ("search", "Does search include note titles?",
     "Yes. Search covers note titles, body text, and text inside attachments like PDFs and images."),
    # shortcuts
    ("shortcut", "How do I make a new note from the keyboard?",
     "Press Ctrl+N, or Cmd+N on Mac. Press Ctrl+/ to see the full list of shortcuts."),
    ("shortcut", "Where can I see every keyboard shortcut?",
     "Press Ctrl+/ (Cmd+/ on Mac) to open the full shortcut list."),
    ("shortcut", "Do shortcuts work on Mac?",
     "Yes, use Cmd instead of Ctrl. For example Cmd+N makes a new note and Cmd+F searches."),
    # billing
    ("billing", "Is Nimbus free?",
     "Yes, there's a free plan. Nimbus Pro costs 4.99 per month or 49 per year and adds bigger attachments, 90 days of version history, unlimited offline devices and priority support."),
    ("billing", "How much is Pro per year?",
     "49 per year, or 4.99 if you pay monthly. The yearly plan works out cheaper."),
    ("billing", "Do I need a credit card for the Pro trial?",
     "No. The 14-day Pro trial needs no card. Start it from Settings > Account > Try Pro."),
    ("billing", "What happens when my Pro trial ends?",
     "You drop back to the free plan automatically unless you subscribe. Nothing is charged silently, and your notes stay exactly where they are."),
    ("billing", "Will I lose my notes if I cancel Pro?",
     "No. Your notes are never deleted when you downgrade. You keep Pro features until the end of the paid period, then your account moves to the free plan."),
    ("billing", "Where do I manage my subscription?",
     "Go to Settings > Account > Subscription. You can see your plan there and cancel if you need to."),
    ("billing", "Does Pro come with better support?",
     "Yes, Nimbus Pro includes priority support, along with 1 GB attachments, 90 days of version history and unlimited offline devices."),
    # attachments
    ("attach", "Why can't I attach a 40 MB video?",
     "Free accounts can attach files up to 25 MB each. Nimbus Pro raises the limit to 1 GB per file, which covers a 40 MB video easily."),
    ("attach", "How many attachments can one note have?",
     "As many as you like. There's no limit on the number of attachments per note, only on the size of each file: 25 MB free, 1 GB on Pro."),
    ("attach", "Can I preview PDFs inside a note?",
     "Yes. Images and PDFs get inline previews and their text is searchable. Other file types show as a download chip."),
    ("attach", "Can I attach a spreadsheet?",
     "Yes, you can attach any file type. Files that aren't images or PDFs show as a download chip inside the note."),
    # history
    ("history", "How far back does version history go?",
     "7 days on the free plan and 90 days on Pro. Open a note, tap the ... menu, and choose Version history to browse it."),
    ("history", "If I restore an old version, do I lose the current one?",
     "No. Nimbus saves the current version into history before restoring, so restoring never loses anything."),
    ("history", "I overwrote a paragraph yesterday. Can I get it back?",
     "Yes. Open the note, tap the ... menu, choose Version history, pick yesterday's version and tap Restore. Free accounts keep 7 days of history."),
    ("history", "Does version history cost extra?",
     "No. Every account has version history: 7 days on the free plan, 90 days on Pro."),
    # templates
    ("template", "How do I start a note from a template?",
     "Tap New note > From template and choose the template you want."),
    ("template", "How do I save a meeting note format to reuse?",
     "Write the note the way you want it, tap the ... menu, and choose Save as template. It then shows up under New note > From template."),
    # reminders
    ("reminder", "Will a reminder show up on all my devices?",
     "Yes. Reminders notify every device you're signed into. Set one by opening the note and tapping the bell icon."),
    ("reminder", "Can I get a reminder every Monday?",
     "Yes. Open the note, tap the bell icon, pick the first Monday and a time, then tap Repeat and choose weekly."),
    ("reminder", "How do I add a reminder to a note?",
     "Open the note, tap the bell icon, and choose a date and time. You'll get a notification on every signed-in device."),
    # widgets
    ("widget", "How do I add Nimbus to my home screen?",
     "Long-press your home screen, add the Nimbus widget, and choose Quick note or a pinned-notes list. It works on iOS and Android."),
    ("widget", "Can the widget show my pinned notes?",
     "Yes. When you add the Nimbus widget, pick the pinned-notes list option instead of Quick note."),
    # security
    ("security", "How do I lock a notebook with Face ID?",
     "Open the notebook, go to Share & security, and turn on Notebook lock. Choose fingerprint or face unlock, or a passcode."),
    ("security", "Do locked notebooks show up in search?",
     "Locked notebooks are excluded from search previews, so their contents don't show in search results until you unlock them."),
    ("security", "Is my data encrypted?",
     "Yes. Notes are encrypted in transit with TLS and at rest on Nimbus servers. You can also lock individual notebooks with a passcode or biometrics."),
    ("security", "Can support see what's in my notes?",
     "No. Support staff can only see note metadata, like counts and sizes, when you open a support ticket, never the contents of your notes."),
    # account
    ("account", "I want to use a different email for Nimbus.",
     "Go to Settings > Account > Email, enter the new address, and confirm it through the link Nimbus sends. Your notes and sharing settings stay the same."),
    ("account", "If I delete my account, can I change my mind?",
     "Yes, for 14 days. Deletion has a 14-day grace period, and signing back in during that time cancels it."),
    ("account", "What happens to my notes when I delete my account?",
     "Nimbus asks you to export your data first, then deletes everything after a 14-day grace period."),
    # import
    ("import", "Can I bring my notes over from Evernote?",
     "Yes. Export them from Evernote as .enex, then go to Settings > Data > Import. Folders become notebooks and hashtags become tags."),
    ("import", "Can I import a folder of Markdown files?",
     "Yes. Settings > Data > Import accepts Markdown and plain text files. The folder structure becomes notebooks."),
    ("import", "Which formats can I import?",
     "Markdown, plain text, and Evernote .enex exports, all from Settings > Data > Import."),
    # markdown
    ("markdown", "How do I make a heading?",
     "Type # followed by a space at the start of a line. Nimbus renders Markdown live, so it becomes a heading straight away."),
    ("markdown", "Can I add code blocks to a note?",
     "Yes. Type three backticks to start a code block. Nimbus renders Markdown live as you type."),
    ("markdown", "I don't know Markdown. Can I still format text?",
     "Yes. Use the formatting toolbar to tap bold, lists, headings and links instead of typing Markdown."),
    ("markdown", "How do I add a link to a note?",
     "Type it in Markdown as [text](url), or select the text and use the link button on the formatting toolbar."),
]


def main() -> None:
    rng = random.Random(7)
    rows = []
    ticket_no = 1000
    for tag, q, a in BASE + PARAPHRASES + MORE:
        rows.append({"id": f"NBS-{ticket_no}", "tag": tag, "question": q, "answer": a})
        ticket_no += 1

    # Dirt, deliberately:
    # 1) exact duplicates, since support exports always have these
    for row in rng.sample(rows, 8):
        rows.append({**row, "id": f"NBS-{ticket_no}"})
        ticket_no += 1
    # 2) answers too short to teach anything
    for q in ["Does Nimbus have dark mode?", "Is Nimbus available on Linux?",
              "Can I use Nimbus in German?", "Does Pro include cloud backup?",
              "Is there a web version?"]:
        rows.append({"id": f"NBS-{ticket_no}", "tag": "misc", "question": q, "answer": rng.choice(["Yes.", "Yes!", "No."])})
        ticket_no += 1

    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 3) malformed lines, which ingest should drop these without dying
        f.write("{this is not json}\n")
        f.write(json.dumps({"id": "NBS-9998", "tag": "broken", "question": "   ", "answer": "Missing question."}) + "\n")
        f.write(json.dumps({"id": "NBS-9999", "tag": "broken", "question": "Missing answer?"}) + "\n")
    print(f"Wrote {len(rows) + 3} lines ({len(rows)} valid tickets) to {OUT}")


if __name__ == "__main__":
    main()
