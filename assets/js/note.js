// Add each new note slug here before linking it from the homepage.
const availableNotes = new Set([
  'gan-learning-notes',
  'medical-image-segmentation',
  'geometry-linear-algebra'
]);

const noteRoot = document.querySelector('#note');
const slug = new URLSearchParams(window.location.search).get('note');

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[character]));
}

// Intentionally small Markdown renderer for locally authored study notes.
function inline(text) {
  return escapeHtml(text)
    .replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, '<img src="$2" alt="$1">')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" rel="noreferrer" target="_blank">$1</a>');
}

function markdownToHtml(markdown) {
  const lines = markdown.replace(/\r/g, '').split('\n');
  const output = [];
  let list = null;
  let code = false;
  const closeList = () => { if (list) { output.push(`</${list}>`); list = null; } };
  lines.forEach((line) => {
    if (line.startsWith('```')) { closeList(); code = !code; output.push(code ? '<pre><code>' : '</code></pre>'); return; }
    if (code) { output.push(`${escapeHtml(line)}\n`); return; }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) { closeList(); const level = heading[1].length; output.push(`<h${level}>${inline(heading[2])}</h${level}>`); return; }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const numbered = line.match(/^\d+\.\s+(.+)$/);
    if (bullet || numbered) { const type = bullet ? 'ul' : 'ol'; if (list !== type) { closeList(); list = type; output.push(`<${type}>`); } output.push(`<li>${inline((bullet || numbered)[1])}</li>`); return; }
    closeList();
    if (!line.trim()) return;
    if (line.startsWith('> ')) output.push(`<blockquote><p>${inline(line.slice(2))}</p></blockquote>`);
    else output.push(`<p>${inline(line)}</p>`);
  });
  closeList();
  return output.join('\n');
}

if (!slug || !availableNotes.has(slug)) {
  noteRoot.innerHTML = '<div class="note-error"><h2>Note not found</h2><p>Please return to the Study Blog and choose an available note.</p></div>';
} else {
  fetch(`${slug}.md`)
    .then((response) => { if (!response.ok) throw new Error('Missing note'); return response.text(); })
    .then((markdown) => {
      noteRoot.innerHTML = markdownToHtml(markdown);
      const title = noteRoot.querySelector('h1');
      if (title) document.title = `${title.textContent} | Sarah Jing`;
    })
    .catch(() => { noteRoot.innerHTML = '<div class="note-error"><h2>Unable to load this note</h2><p>Try opening the website through a local server or GitHub Pages.</p></div>'; });
}
