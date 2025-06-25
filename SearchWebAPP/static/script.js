let currentQuestion = '';
let additionalQuestion = '';
let chatHistory = [];

function appendChatMessage(role, message) {
    const chatHistoryDiv = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    messageDiv.textContent = message;
    chatHistoryDiv.prepend(messageDiv);
    
    // チャット履歴を配列に保存
    chatHistory.push({
        role: role,
        message: message
    });
}

async function sendChat() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    if (!message) return;

    appendChatMessage('user', message);

    currentQuestion = message;
    showLoading();
    clearResults();

    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message,
                chat_history: chatHistory,
                top_n: 10
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        hideLoading();

        // LLMからの応答を表示
        if (data.target || data.service_description) {
            let llmResponse = '';
            if (data.target) {
                llmResponse += `【ユーザー像】\n${data.target}\n`;
                if (data.target_labels && data.target_labels.length > 0) {
                    llmResponse += `対象者ラベル: ${data.target_labels.join(', ')}\n`;
                }
            }
            if (data.service_description) {
                llmResponse += `\n【探しているサービス内容】\n${data.service_description}\n`;
                if (data.service_labels && data.service_labels.length > 0) {
                    llmResponse += `サービスラベル: ${data.service_labels.join(', ')}\n`;
                }
            }
            appendChatMessage('agent', llmResponse);
        }

        if (!data.has_sufficient_info) {
            if (data.additional_question && data.additional_question !== 'confirmed') {
                appendChatMessage('agent', data.additional_question);
            }
            return;
        }

        appendChatMessage('agent', '該当するサービスをリストアップしました。');
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        hideLoading();
        appendChatMessage('agent', '申し訳ありません。エラーが発生しました。もう一度お試しください。');
    }

    chatInput.value = '';
}

async function submitAdditionalInfo() {
    const additionalInfo = document.getElementById('additionalInput').value.trim();
    if (!additionalInfo) return;

    appendChatMessage('user', additionalInfo);

    const combinedQuestion = `${currentQuestion}
${additionalQuestion}
${additionalInfo}`;
    document.getElementById('chatInput').value = combinedQuestion;
    await sendChat();
}

function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

function showAdditionalQuestion(question) {
    additionalQuestion = question;
    const questionElement = document.getElementById('additionalQuestionText');
    questionElement.textContent = question;
    document.getElementById('additionalQuestion').style.display = 'block';
    document.getElementById('additionalInput').value = '';
}

function hideAdditionalQuestion() {
    document.getElementById('additionalQuestion').style.display = 'none';
}

function clearResults() {
    document.getElementById('results').innerHTML = '';
    document.getElementById('userLabels').innerHTML = '';
    document.getElementById('userProfile').textContent = '';
    document.getElementById('serviceDescription').textContent = '';
    // チャット履歴はクリアしない
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    const userLabelsDiv = document.getElementById('userLabels');
    const userProfileDiv = document.getElementById('userProfile');
    const serviceDescriptionDiv = document.getElementById('serviceDescription');
    
    // ユーザー情報の表示
    if (data.target || data.service_description) {
        // ユーザー像の表示
        let userProfileHtml = '<h4>ユーザー像</h4>';
        userProfileHtml += `<p>${data.target || '情報なし'}</p>`;
        if (data.target_labels && data.target_labels.length > 0) {
            userProfileHtml += '<div class="labels-container">';
            data.target_labels.forEach(label => {
                userProfileHtml += `<span class="label target-label">${label}</span>`;
            });
            userProfileHtml += '</div>';
        }
        userProfileDiv.innerHTML = userProfileHtml;
        
        // サービス内容の表示
        let serviceDescriptionHtml = '<h4>探しているサービス内容</h4>';
        serviceDescriptionHtml += `<p>${data.service_description || '情報なし'}</p>`;
        if (data.service_labels && data.service_labels.length > 0) {
            serviceDescriptionHtml += '<div class="labels-container">';
            data.service_labels.forEach(label => {
                serviceDescriptionHtml += `<span class="label service-label">${label}</span>`;
            });
            serviceDescriptionHtml += '</div>';
        }
        serviceDescriptionDiv.innerHTML = serviceDescriptionHtml;
    }

    // 検索結果の表示
    resultsDiv.innerHTML = '';
    if (data.results && data.results.length > 0) {
        data.results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'result-item';

            let labelsHtml = '';
            const matchInfo = result.match_info;
            if (matchInfo) {
                if (matchInfo.matched_target_labels) {
                    matchInfo.matched_target_labels.forEach(label => {
                        labelsHtml += `<span class="label target-label">${label}</span>`;
                    });
                }
                if (matchInfo.matched_service_labels) {
                    matchInfo.matched_service_labels.forEach(label => {
                        labelsHtml += `<span class="label service-label">${label}</span>`;
                    });
                }
            }

            resultItem.innerHTML = `
                <h3>${result.formal_name}</h3>
                <p>${result.overview}</p>
                <div class="result-labels">${labelsHtml}</div>
                <p><a href="${result.url}" target="_blank">詳細を見る</a></p>
                <p class="score">類似度: ${(result.score * 100).toFixed(1)}%</p>
            `;
            resultsDiv.appendChild(resultItem);
        });
    } else {
        resultsDiv.innerHTML = '<p class="no-results">該当するサービスが見つかりませんでした。</p>';
    }
}

document.getElementById('chatInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendChat();
    }
});

// 送信ボタンのクリックイベントハンドラを追加
document.getElementById('sendButton').addEventListener('click', function() {
    sendChat();
});

document.getElementById('additionalInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        submitAdditionalInfo();
    }
});