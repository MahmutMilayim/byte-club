/* ==================================
   IFAB Rules Database
   Laws of the Game 2024/25
================================== */

const IFABRules = {
    // Law 11 - Offside
    law11: {
        name: "Law 11 - Ofsayt",
        fullName: "Ofsayt (Offside)",
        description: "Bir oyuncu, rakip yarı sahada, toptan ve sondan bir önceki rakip oyuncudan daha yakınsa ofsayt pozisyonundadır.",
        sections: {
            position: {
                title: "Ofsayt Pozisyonu",
                content: `Bir oyuncu, başı, gövdesi veya ayakları rakip yarı sahada (orta çizgi hariç) ve toptan ile sondan bir önceki rakip oyuncudan daha yakınsa ofsayt pozisyonundadır.

Kollar ve eller bu değerlendirmeye dahil değildir.

Ofsayt pozisyonunda olmak tek başına ihlal değildir.`
            },
            offence: {
                title: "Ofsayt İhlali",
                content: `Ofsayt pozisyonunda olan oyuncu, takım arkadaşı tarafından oynanan veya dokunulan an itibarıyla:
• Oyuna müdahale ederse (topa oynar veya dokunursa)
• Rakibe müdahale ederse (rakibin topa oynamasını veya topu oynama olasılığını engellerse)
• Ofsayt pozisyonunda bulunarak avantaj sağlarsa`
            },
            noOffence: {
                title: "İhlal Olmayan Durumlar",
                content: `Oyuncu, topu doğrudan şu atışlardan alırsa ofsayt ihlali yoktur:
• Taç atışı
• Kale vuruşu
• Korner vuruşu

Ayrıca, rakip oyuncu kasıtlı olarak topu oynarsa (kurtarış hariç) ofsayt ihlali yoktur.`
            }
        },
        keywords: ["ofsayt", "offside", "çizgi", "pozisyon", "aktif", "pasif"]
    },

    // Law 12 - General
    law12: {
        name: "Law 12 - Fauller",
        fullName: "Fauller ve Kötü Davranış",
        description: "Doğrudan veya dolaylı serbest vuruşla cezalandırılan ihlaller ve disiplin tedbirleri.",
        sections: {
            directFK: {
                title: "Doğrudan Serbest Vuruş",
                content: `Aşağıdaki ihlaller doğrudan serbest vuruşla cezalandırılır:
• Rakibe çelme takmak veya takmaya teşebbüs
• Rakibe atlamak
• Rakibi itmek
• Rakibe vurmak veya vurmaya teşebbüs
• Rakibe tekme atmak veya atmaya teşebbüs
• Rakibi tutmak
• Aşırı güç veya vahşilikle oynamak

Temas dikkatsiz, tedbirsiz veya aşırı güç kullanarak olabilir.`
            },
            careless: {
                title: "Temas Seviyeleri",
                content: `• Dikkatsiz (Careless): Oyuncu dikkatsiz veya ihtiyatsız davranır - Faul
• Tedbirsiz (Reckless): Oyuncu rakibin güvenliğini tehlikeye atar - Sarı kart
• Aşırı Güç (Excessive Force): Oyuncu gerekenden çok fazla güç kullanır - Kırmızı kart`
            }
        },
        keywords: ["faul", "itme", "çelme", "tekme", "temas"]
    },

    // Law 12 - Handball
    "law12-handball": {
        name: "Law 12 - El",
        fullName: "El ile Oynama (Handball)",
        description: "Elin/kolun topla teması ve hangi durumlarda ihlal oluştuğuna dair kurallar.",
        sections: {
            offence: {
                title: "İhlal Olan Durumlar",
                content: `El ile oynama ihlali:
• Oyuncu, topu eli veya kolu ile kasıtlı olarak oynarsa
• Oyuncu, vücudunu el veya kolu ile doğal olmayan şekilde büyütürse
• Oyuncu, eli veya kolu omuz seviyesinin üzerinde olacak şekilde topa el veya kolla dokunursa

Kolun vücuttaki pozisyonu önemlidir. Vücuttan uzaklaştırılmış kol ile temas genellikle ihlaldir.`
            },
            noOffence: {
                title: "İhlal Olmayan Durumlar",
                content: `El ile oynama ihlali değildir:
• Top, oyuncunun kendi başından veya vücudundan sekerek eline değerse
• Top, yakın mesafedeki başka bir oyuncudan sekerek eline değerse
• Kol vücuda yakın ve doğal pozisyonda ise
• Oyuncu düşerken ve kollarını destek için kullanırken temas olursa`
            },
            attackingPlayer: {
                title: "Hücum Oyuncusu",
                content: `Hücum eden takım oyuncusunun (veya takım arkadaşının) eli veya kolu ile temastan sonra:
• Top doğrudan kaleye girerse - Gol iptal
• Gol şansı oluşturulursa - Serbest vuruş

Bu durumlarda kasıt aranmaz; her türlü el teması ihlaldir.`
            }
        },
        keywords: ["el", "handball", "kol", "el pozisyonu", "doğal pozisyon"]
    },

    // Law 12 - Reckless/Serious Foul Play
    "law12-reckless": {
        name: "Law 12 - Sert Faul",
        fullName: "Tedbirsiz ve Kontrolsüz Müdahale",
        description: "Sarı ve kırmızı kartla cezalandırılan tehlikeli müdahaleler.",
        sections: {
            reckless: {
                title: "Tedbirsiz Müdahale (Sarı Kart)",
                content: `Tedbirsiz müdahale, oyuncunun rakibinin güvenliğini tehlikeye attığı durumlarda yapılan müdahaledir.

Sarı kart gerektiren durumlar:
• Oyunun akışını engelleyen faul
• Hakem kararına itiraz
• Rakibin umut verici bir ataği durdurma
• Provokatif hareketler`
            },
            excessive: {
                title: "Aşırı Güç Kullanımı (Kırmızı Kart)",
                content: `Ciddi faul oyunu (Serious Foul Play), oyuncunun rakibinin güvenliğini tehlikeye atan, aşırı güç veya vahşet içeren top mücadelelesidir.

Kırmızı kart gerektiren durumlar:
• Top için yapılan mücadelede aşırı güç kullanımı
• Vahşi müdahale
• Rakibin fiziksel bütünlüğünü tehlikeye atan hareketler`
            },
            studsUp: {
                title: "Krampon Gösterme",
                content: `Kayarak veya ayakta müdahalede:
• Krampon yukarıda
• Bacak düz ve sert
• Dizi bükülmemiş

şekilde yapılan müdahaleler genellikle aşırı güç olarak değerlendirilir ve kırmızı kartla cezalandırılır.`
            }
        },
        keywords: ["sert faul", "krampon", "tedbirsiz", "aşırı güç", "vahşet"]
    },

    // Law 12 - DOGSO
    "law12-dogso": {
        name: "Law 12 - DOGSO",
        fullName: "Bariz Gol Şansının Engellenmesi",
        description: "Denying an Obvious Goal-Scoring Opportunity (DOGSO) - En ağır disiplin ihlallerinden biri.",
        sections: {
            definition: {
                title: "DOGSO Kriterleri",
                content: `Bariz gol şansının engellenmesi için 4 kriter değerlendirilir:
1. Kaleye olan mesafe
2. Topun kontrolü ve oynanabilirliği
3. Savunma oyuncularının sayısı ve pozisyonu
4. Oyunun genel akışı

Tüm kriterlerin birlikte değerlendirilmesi gerekir.`
            },
            foulInBox: {
                title: "Ceza Sahasında DOGSO",
                content: `Ceza sahası içinde, rakibin bariz gol şansını faulle engelleyen oyuncu:
• Eğer faulü "top için" yapmış ve "meşru bir teşebbüste" bulunmuşsa - Sarı kart + Penaltı
• Eğer topu oynama teşebbüsü yoksa - Kırmızı kart + Penaltı
• Tutma, çekme, itme gibi hareketler - Kırmızı kart + Penaltı`
            },
            outsideBox: {
                title: "Ceza Sahası Dışında DOGSO",
                content: `Ceza sahası dışında, rakibin bariz gol şansını faulle engelleyen oyuncu:
• Kırmızı kart ile cezalandırılır
• Serbest vuruş verilir

Kaleci dahil tüm oyuncular için geçerlidir.`
            }
        },
        keywords: ["DOGSO", "bariz gol şansı", "son adam", "penaltı", "kırmızı kart"]
    },

    // Law 14 - Penalty Kick
    law14: {
        name: "Law 14 - Penaltı",
        fullName: "Penaltı Vuruşu",
        description: "Ceza sahası içinde yapılan ihlallerde verilen ve özel kurallara tabi vuruş.",
        sections: {
            award: {
                title: "Penaltı Verilme Koşulları",
                content: `Savunma yapan takımın oyuncusu, kendi ceza sahası içinde ve top oyundayken:
• Doğrudan serbest vuruş gerektiren bir faul yaparsa
• El ile oynama ihlali yaparsa

penaltı vuruşu verilir.

Topun konumu önemli değildir, ihlalin yapıldığı yer önemlidir.`
            },
            execution: {
                title: "Penaltı Vuruşu Kuralları",
                content: `• Top penaltı noktasına sabit olarak konur
• Kaleci kale çizgisinde olmalıdır
• Diğer oyuncular ceza sahası dışında olmalıdır
• Vuruş yapılana kadar top hareketsiz olmalıdır`
            },
            goalkeepers: {
                title: "Kaleci Kuralları",
                content: `Penaltı atışında kaleci:
• Vuruş yapılana kadar iki ayağından en az biri kale çizgisine değmeli veya çizgi üzerinde olmalı
• Erkenden hareket ederse ve gol olmazsa - Tekrar
• Erkenden hareket ederse ve gol olursa - Gol geçerli`
            }
        },
        keywords: ["penaltı", "penalty", "ceza sahası", "on bir metre"]
    }
};

// Rule Helper Functions
function getRuleInfo(ruleKey) {
    return IFABRules[ruleKey] || null;
}

function getRuleSummary(ruleKey) {
    const rule = IFABRules[ruleKey];
    if (!rule) return null;
    return {
        name: rule.name,
        description: rule.description
    };
}

function searchRulesByKeyword(keyword) {
    const results = [];
    const lowerKeyword = keyword.toLowerCase();
    
    for (const [key, rule] of Object.entries(IFABRules)) {
        if (rule.keywords.some(k => k.includes(lowerKeyword))) {
            results.push({ key, rule });
        }
    }
    
    return results;
}

function detectRelevantRules(text) {
    const lowerText = text.toLowerCase();
    const detected = [];
    
    // Keyword mapping for auto-detection
    const keywordMap = {
        'law11': ['ofsayt', 'offside', 'çizgi', 'son savunmacı', 'aktif pozisyon'],
        'law12': ['faul', 'temas', 'itme', 'çelme', 'tekme'],
        'law12-handball': ['el', 'hand', 'kol', 'el pozisyonu', 'handball'],
        'law12-reckless': ['sert', 'krampon', 'tehlikeli', 'kontrolsüz', 'vahşi'],
        'law12-dogso': ['dogso', 'bariz gol', 'son adam', 'gol şansı'],
        'law14': ['penaltı', 'penalty', 'ceza sahası', 'on bir', 'penaltı noktası']
    };
    
    for (const [ruleKey, keywords] of Object.entries(keywordMap)) {
        for (const keyword of keywords) {
            if (lowerText.includes(keyword)) {
                if (!detected.includes(ruleKey)) {
                    detected.push(ruleKey);
                }
                break;
            }
        }
    }
    
    return detected;
}

// Decision Templates
const DecisionTemplates = {
    penalty: {
        decision: "PENALTI",
        icon: "⚽",
        color: "primary"
    },
    noPenalty: {
        decision: "PENALTI YOK - DEVAM",
        icon: "▶️",
        color: "success"
    },
    foul: {
        decision: "FAUL",
        icon: "🚩",
        color: "warning"
    },
    offside: {
        decision: "OFSAYT",
        icon: "🚩",
        color: "warning"
    },
    noOffside: {
        decision: "OFSAYT YOK",
        icon: "✅",
        color: "success"
    },
    yellowCard: {
        decision: "SARI KART",
        icon: "🟨",
        color: "warning"
    },
    redCard: {
        decision: "KIRMIZI KART",
        icon: "🟥",
        color: "danger"
    },
    play_on: {
        decision: "DEVAM",
        icon: "▶️",
        color: "success"
    },
    goal: {
        decision: "GOL GEÇERLİ",
        icon: "⚽",
        color: "success"
    },
    noGoal: {
        decision: "GOL İPTAL",
        icon: "❌",
        color: "danger"
    }
};

// Export for use in app.js
window.IFABRules = IFABRules;
window.getRuleInfo = getRuleInfo;
window.getRuleSummary = getRuleSummary;
window.searchRulesByKeyword = searchRulesByKeyword;
window.detectRelevantRules = detectRelevantRules;
window.DecisionTemplates = DecisionTemplates;
