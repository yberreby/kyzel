# Glossary

- A **conversation** is a sequence of (ChatML) messages. It's a flat stream, or list.
- An **event** is "something that happened during a session"
  - Examples: the user sent a message to the model; the model sent a message to the user; the model thought to itself; the model requested execution of some code; the execution yielded some result...
  - One could consider less obvious events: the user *preferred* such-and-such generation over this other one; the user reworded the model's output
  - Importantly, the user can choose to roll back to an earlier point in the conversation (not necessarily perfectly, but at least roll back the chat history), and they do so through relevant events.
- A **session** is the record-to-date of a single series of back-and-forths with the model. It may contain many events, as a chronologically-ordered list, but it implicitly can branch out into many conversation paths, potential or actual.
