const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const sidebar = document.getElementById('sidebar');
const toggleSidebar = document.getElementById('toggle-sidebar');
const addConversation = document.getElementById('add-conversation');
const conversationList = document.getElementById('conversation-list');

let selectedConversationId = null;
let conversationHistory = {}; // id -> [{sender, text}]
const token = localStorage.getItem('token');
const BASE_URL = 'http://192.168.0.89:8000';

if (!token) {
    alert('請先登入');
    window.location.href = '/static/login.html';
}

function createConversationElement(title, id) {
    const item = document.createElement('li');
    item.dataset.conversationId = id;
    item.innerHTML = `
        <span class="conversation-title">${title}</span>
        <button class="rename-conversation">✎</button>
        <button class="delete-conversation">×</button>
    `;
    conversationList.insertBefore(item, conversationList.firstChild);

    item.addEventListener('click', async () => {
        await loadConversation(id);
        selectConversation(item);
    });

    item.querySelector('.rename-conversation').addEventListener('click', async (e) => {
        e.stopPropagation();
        const newTitle = prompt("輸入新聊天室名稱：");
        if (!newTitle) return;
        const res = await fetch(`${BASE_URL}/conversation/${id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ title: newTitle })
        });
        if (res.ok) item.querySelector('.conversation-title').textContent = newTitle;
    });

    item.querySelector('.delete-conversation').addEventListener('click', async (e) => {
        e.stopPropagation();
        await fetch(`${BASE_URL}/conversation/${id}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });
        delete conversationHistory[id];
        if (selectedConversationId === id) {
            chatMessages.innerHTML = '';
            selectedConversationId = null;
        }
        item.remove();
    });

    return item;
}

function selectConversation(item) {
    const id = item.dataset.conversationId;
    if (selectedConversationId === id) return;
    document.querySelectorAll('#conversation-list li').forEach(li => li.classList.remove('selected'));
    item.classList.add('selected');
    selectedConversationId = id;
    renderConversation();
}

function renderConversation() {
    chatMessages.innerHTML = '';
    const msgs = conversationHistory[selectedConversationId] || [];
    msgs.forEach(msg => appendMessage(msg.sender, msg.text));
}

function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.classList.add(`${sender}-message`);
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    if (selectedConversationId) {
        if (!conversationHistory[selectedConversationId]) conversationHistory[selectedConversationId] = [];
        conversationHistory[selectedConversationId].push({ sender, text });
    }
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || !selectedConversationId) return;
    appendMessage('user', text);
    userInput.value = '';

    const res = await fetch(`${BASE_URL}/chat/${selectedConversationId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ question: text })
    });
    const data = await res.json();
    appendMessage('bot', data.reply);

    // ✅ 若有 updated_title，更新左側標題文字
    if (data.updated_title && selectedConversationId) {
        const li = document.querySelector(`[data-conversation-id="${selectedConversationId}"]`);
        if (li) {
            const span = li.querySelector('.conversation-title');
            if (span) span.textContent = data.updated_title;
        }
    }
}

async function loadAllConversations() {
    const res = await fetch(`${BASE_URL}/conversations`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    const data = await res.json();
    data.forEach(conv => {
        const item = createConversationElement(conv.title, conv.id);
        conversationHistory[conv.id] = [];
        conv.messages.forEach(m => {
            conversationHistory[conv.id].push({ sender: 'user', text: m.question });
            conversationHistory[conv.id].push({ sender: 'bot', text: m.answer });
        });
    });
}

async function loadConversation(id) {
    const res = await fetch(`${BASE_URL}/conversation/${id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });
    const data = await res.json();
    conversationHistory[id] = [];
    data.messages.forEach(m => {
        conversationHistory[id].push({ sender: 'user', text: m.question });
        conversationHistory[id].push({ sender: 'bot', text: m.answer });
    });
    renderConversation();
}

async function createNewConversation() {
    const title = prompt('請輸入新聊天室名稱：');
    if (!title) return;
    const res = await fetch(`${BASE_URL}/start_conversation`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ title })
    });
    const data = await res.json();
    createConversationElement(title, data.conversation_id);
}

document.addEventListener('DOMContentLoaded', async () => {
    toggleSidebar?.addEventListener('click', () => {
        sidebar.classList.toggle('expanded');
    });

    addConversation?.addEventListener('click', createNewConversation);

    sendButton?.addEventListener('click', sendMessage);
    userInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    await loadAllConversations();
    // 如果網址中有 ?conversation_id=xxx，就自動載入該聊天室
    const urlParams = new URLSearchParams(window.location.search);
    const autoLoadId = urlParams.get("conversation_id");
    if (autoLoadId) {
        loadConversation(autoLoadId).then(() => {
            const item = document.querySelector(`[data-conversation-id="${autoLoadId}"]`);
            if (item) selectConversation(item);
        });
    }

});


