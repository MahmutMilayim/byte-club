/* ==================================
   Football Rules Data for RAG System
   IFAB Laws of the Game 2024/25
   Auto-loaded into RAG System
================================== */

const FootballRulesData = [
    {
        name: "Law 11 - Ofsayt Kuralları.txt",
        content: `IFAB FUTBOL OYUN KURALLARI - LAW 11: OFSAYT

OFSAYT POZİSYONU
================
Bir oyuncu, başı, gövdesi veya ayakları rakip yarı sahada (orta çizgi hariç) ve toptan ile sondan bir önceki rakip oyuncudan daha yakınsa ofsayt pozisyonundadır.

Kollar ve eller bu değerlendirmeye dahil değildir.
Ofsayt pozisyonunda olmak tek başına ihlal değildir.

OFSAYT İHLALİ
=============
Ofsayt pozisyonunda olan oyuncu, takım arkadaşı tarafından oynanan veya dokunulan an itibarıyla:
• Oyuna müdahale ederse (topa oynar veya dokunursa)
• Rakibe müdahale ederse (rakibin topa oynamasını veya topu oynama olasılığını engellerse)
• Ofsayt pozisyonunda bulunarak avantaj sağlarsa (top kaleciden veya direk/çıtadan sekerek gelirse)

Oyuna müdahale, oyuncunun takım arkadaşı tarafından oynanan veya dokunulan topa:
- Oynaması veya dokunması
- Topla temas etmesi

Rakibe müdahale:
- Rakibin görüş alanını kapatarak topa oynamasını engelleme
- Rakiple top için mücadele etme
- Rakibin topa ulaşma girişimini engelleme
- Topun yakınında durup rakibin topa oynama kararını etkileme

Avantaj sağlama:
- Ofsayt pozisyonundayken kalecinin kurtarışından veya direk/çıtadan dönen topu oynama
- Kasıtsız olarak rakipten gelen topu ofsayt pozisyonunda oynama

İHLAL OLMAYAN DURUMLAR
======================
Oyuncu, topu doğrudan şu atışlardan alırsa ofsayt ihlali yoktur:
• Taç atışı
• Kale vuruşu  
• Korner vuruşu

Ayrıca, rakip oyuncu kasıtlı olarak topu oynarsa (kurtarış hariç) ofsayt ihlali yoktur.

CEZA
====
Ofsayt ihlali olduğunda, hakem ihlalin gerçekleştiği yerden dolaylı serbest vuruş verir.

VAR KULLANIMI
=============
VAR, ofsayt çizgisi çizilerek milimetrik ölçümler yapabilir.
Yarı otomatik ofsayt sistemi (SAOT) kullanılabilir.
Vücut hatları kullanılarak kesin tespit yapılır.`
    },
    {
        name: "Law 12 - Fauller ve Kötü Davranış.txt",
        content: `IFAB FUTBOL OYUN KURALLARI - LAW 12: FAULLER VE KÖTÜ DAVRANIŞ

DOĞRUDAN SERBEST VURUŞ GEREKTİREN İHLALLER
==========================================
Aşağıdaki ihlaller doğrudan serbest vuruşla cezalandırılır:
• Rakibe çelme takmak veya takmaya teşebbüs etmek
• Rakibe atlamak
• Rakibe vurmak veya vurmaya teşebbüs etmek (kafa ile dahil)
• Rakibi itmek
• Rakibe tekme atmak veya atmaya teşebbüs etmek
• Rakibi engellemek (temas ile)
• Rakibi tutmak
• Rakibe tükürmek veya tükürmeye teşebbüs etmek
• Topu elle oynamak (kalecinin kendi ceza sahası hariç)

TEMAS SEVİYELERİ
================
DİKKATSİZ (Careless): 
- Oyuncu rakibe müdahale ederken dikkatsiz veya ihtiyatsız davranır
- Sadece faul cezası verilir
- Kart gösterilmez

TEDBİRSİZ (Reckless):
- Oyuncu, rakibinin güvenliğini dikkate almadan müdahale eder
- SARI KART gösterilir
- Oyunun tehlikeye atılması

AŞIRI GÜÇ (Excessive Force):
- Oyuncu gerekenden çok fazla güç kullanır
- Rakibin fiziksel bütünlüğünü tehlikeye atar
- KIRMIZI KART gösterilir

SARI KART GEREKTIREN DURUMLAR
=============================
• Sportmenlik dışı davranış
• Sözle veya hareketle itiraz etme
• Hakemin iznini almadan sahaya girme/çıkma
• Hakem kararlarında belirlenen mesafeye uymama
• Oyunu tekrar tekrar durduran ihlaller
• Oyunun yeniden başlamasını geciktirme
• Korner/serbest vuruşta belirlenen mesafeye uymama
• VAR incelemesi alanına kasıtlı girme

KIRMIZI KART GEREKTIREN DURUMLAR
================================
• Ciddi faul oyunu (Serious Foul Play)
• Şiddet içeren davranış
• Rakibe veya başka birine tükürme
• Bariz gol şansını elleyle engelleme (kaleci hariç)
• Genel olarak kaleye doğru hareket eden bir oyuncunun bariz gol şansını faulle engelleme
• Kaba, hakaret içeren sözler ve/veya hareketler
• İkinci sarı kart

CİDDİ FAUL OYUNU (SERIOUS FOUL PLAY)
====================================
Top için mücadelede veya mücadele için hazırlık yaparken:
- Aşırı güç kullanan
- Vahşilik içeren
- Rakibin güvenliğini tehlikeye atan müdahaleler

Krampon yukarıda, bacak düz ve sert, diz bükülmemiş şekilde yapılan müdahaleler genellikle ciddi faul oyunu olarak değerlendirilir.`
    },
    {
        name: "Law 12 - El ile Oynama (Handball).txt",
        content: `IFAB FUTBOL OYUN KURALLARI - LAW 12: EL İLE OYNAMA (HANDBALL)

EL İLE OYNAMA İHLALİ OLAN DURUMLAR
==================================
Aşağıdaki durumlarda el ile oynama ihlali vardır:

1. KASITLI EL TEMASI:
   - Oyuncu eli veya kolu ile kasıtlı olarak topa dokunursa
   - Eli veya kolu ile topa doğru hareket ederse
   
2. VÜCUDU BÜYÜTME:
   - El veya kol ile vücut doğal olmayan şekilde büyütülürse
   - Kolun pozisyonu, oyuncunun hareketiyle gerekçelendirilemezse
   
3. OMUZ SEVİYESİNİN ÜZERİNDE:
   - El veya kol omuz seviyesinin üzerindeyse (kasıtlı olarak topa oynama veya vücudu büyütme olmasa bile)

KOLUN POZİSYONU ÖNEMLİDİR:
- Kol vücuttan uzaklaştırılmışsa risk artar
- Kol, o durumda beklenen pozisyonun dışındaysa
- Oyuncunun duruşu ve hareketi değerlendirilmeli

EL İLE OYNAMA İHLALİ OLMAYAN DURUMLAR
=====================================
1. Top, oyuncunun kendi başından veya vücudundan (ayak dahil) sekerek eline değerse
2. Top, yakın mesafedeki başka bir oyuncunun başından veya vücudundan sekerek eline değerse
3. El veya kol vücuda yakın ve doğal pozisyondaysa
4. Oyuncu yere düşerken kollarını destek için kullandığında temas olursa (kollar vücuttan yana veya dikey olarak uzatılmamışsa)

HÜCUM OYUNCUSU için ÖZEL KURALLAR
=================================
Aşağıdaki durumlarda HER ZAMAN ihlaldir (kasıt aranmaz):

1. DOĞRUDAN GOL:
   - Hücum oyuncusunun eli/kolu ile temastan sonra top doğrudan kaleye girerse → GOL İPTAL
   
2. GOL ŞANSI OLUŞTURMA:
   - Hücum oyuncusu (veya takım arkadaşı) eli/kolu ile temastan sonra gol şansı oluşturursa → SERBEST VURUŞ
   - Top kontrole alınsa veya pas verilse bile ihlaldir

3. TAKIM ARKADAŞI DAHIL:
   - Golü atan oyuncu olmasa bile, takım arkadaşının eli ile temas olduysa değerlendirilir

KALECİ için KURALLAR
====================
- Kaleci kendi ceza sahası içinde topu elle oynayabilir
- Ceza sahası dışında normal oyuncu kuralları geçerlidir
- Kaleci takım arkadaşından kasıtlı ayak pası alırsa elle tutamaz (6 saniyelik sınırlama ayrıca geçerli)

VAR DEĞERLENDİRMESİ
===================
- El temasının olup olmadığı
- Kolun pozisyonu
- Topun mesafesi ve hızı
- Hücum fazında mı savunma fazında mı
- Gol veya gol şansı ile bağlantısı`
    },
    {
        name: "Law 12 - DOGSO (Bariz Gol Şansı).txt",
        content: `IFAB FUTBOL OYUN KURALLARI - LAW 12: DOGSO
(DENYING AN OBVIOUS GOAL-SCORING OPPORTUNITY)
BARİZ GOL ŞANSININ ENGELLENMESİ

DOGSO KRİTERLERİ
================
Bariz gol şansının engellenmesi için 4 kriter birlikte değerlendirilir:

1. KALEYE MESAFE:
   - Oyuncu kaleye ne kadar yakın?
   - Gol atma olasılığı açısından mesafe uygun mu?

2. TOPUN KONTROLÜ VE OYNANABİLİRLİĞİ:
   - Oyuncu topu kontrol edebilir durumda mı?
   - Top, oyuncunun oynayabileceği mesafede mi?
   
3. SAVUNMA OYUNCULARININ SAYISI VE POZİSYONU:
   - Kaç savunmacı topla kaleci arasında?
   - Savunmacılar müdahale edebilecek konumda mı?
   
4. OYUNUN GENEL AKIŞI:
   - Hücum yönü kaleye doğru mu?
   - Oyun temposu ve akışı gol şansını destekliyor mu?

Tüm kriterlerin birlikte olması gerekir!

CEZA SAHASI İÇİNDE DOGSO
========================
Ceza sahası içinde, rakibin bariz gol şansını faulle engelleyen oyuncu:

SARI KART + PENALTI verilir eğer:
- Faul "top için" yapılmışsa
- "Meşru bir teşebbüs" varsa (oyuncu gerçekten topa oynamaya çalışmış)

KIRMIZI KART + PENALTI verilir eğer:
- Topu oynama teşebbüsü yoksa
- Tutma, çekme, itme gibi hareketler varsa
- Topla hiçbir şekilde temas girişimi yoksa

CEZA SAHASI DIŞINDA DOGSO
=========================
Ceza sahası dışında, rakibin bariz gol şansını faulle engelleyen oyuncu:
• KIRMIZI KART ile cezalandırılır
• Serbest vuruş verilir

Bu kural kaleci dahil tüm oyuncular için geçerlidir.

EL İLE DOGSO
============
Bir oyuncu (kaleci hariç) eli ile bariz gol şansını engellerse:
- KIRMIZI KART
- Penaltı (ceza sahası içinde) veya serbest vuruş (ceza sahası dışında)

DOGSO OLMAYAN DURUMLAR
======================
- Birden fazla savunmacı iyi pozisyondaysa
- Kaleci müdahale edebilecek durumdaysa
- Kaleye mesafe çok fazlaysa
- Top kontrolden çıkmışsa

VAR KULLANIMI
=============
DOGSO, VAR'ın müdahale ettiği kritik kararlardan biridir.
- Faul var mı yok mu?
- DOGSO kriterleri sağlanıyor mu?
- Ceza sahası içinde mi dışında mı?
- Sarı mı kırmızı mı olmalı?`
    },
    {
        name: "Law 14 - Penaltı Kuralları.txt",
        content: `IFAB FUTBOL OYUN KURALLARI - LAW 14: PENALTI VURUŞU

PENALTI VERİLME KOŞULLARI
=========================
Savunma yapan takımın oyuncusu, kendi ceza sahası içinde ve top oyundayken:
• Doğrudan serbest vuruş gerektiren bir faul yaparsa
• El ile oynama ihlali yaparsa

PENALTI VERİLİR.

ÖNEMLİ: Topun konumu değil, ihlalin yapıldığı yer önemlidir!
- İhlal ceza sahası içindeyse = Penaltı
- Top ceza sahası dışında olsa bile = Penaltı (ihlal içerideyse)

PENALTI VURUŞU KURALLARI
========================
TOP:
- Penaltı noktasına sabit olarak konur
- Vuruş yapılana kadar hareketsiz olmalı

VURUŞU YAPACAK OYUNCU:
- Açıkça belirlenmeli
- Sadece ileri doğru tekmeleyebilir
- Vuruştan sonra tekrar topa dokunamaz (başka oyuncu dokunmadan)

DİĞER OYUNCULAR:
- Ceza sahası dışında olmalı
- Penaltı yayının dışında olmalı
- Penaltı noktasının gerisinde olmalı

KALECİ KURALLARI
================
VURUŞ ANINDA:
- İki ayağından en az biri kale çizgisine değmeli veya çizgi üzerinde olmalı

ERKEN HAREKET:
- Kaleci erkenden hareket ederse VE gol olmazsa → TEKRAR
- Kaleci erkenden hareket ederse VE gol olursa → GOL GEÇERLİ

KALE ÇİZGİSİ:
- Kaleci kale çizgisinin önüne geçemez (vuruş anında)
- Hareket yapmadan önce çizgide beklemelidir

İHLALLER VE SONUÇLARI
=====================
VURUŞU YAPAN OYUNCU İHLAL EDERSE:
- Gol olursa → GEÇERSİZ, Dolaylı serbest vuruş
- Gol olmazsa → Dolaylı serbest vuruş

KALECİ İHLAL EDERSE:
- Gol olursa → GEÇERLİ
- Gol olmazsa → TEKRAR

TAKIM ARKADAŞI İHLAL EDERSE:
- Kendi takım arkadaşının ihlali gibi değerlendirilir
- Duruma göre tekrar veya fark sonuç

HER İKİ TAKIM DA İHLAL EDERSE:
- TEKRAR

SEKME (REBOUND)
===============
- Top direkten veya kaleciden sekerse ve oyuna devam ederse:
  • Vuruşu yapan oyuncu dahil herkes topa oynayabilir
  • Vuruşu yapan oyuncu ikinci kez dokunursa → Dolaylı serbest vuruş

VAR ve PENALTI
==============
VAR kontrol edebilir:
- Faul/ihlal ceza sahası içinde mi dışında mı?
- Faul var mı yok mu?
- Kaleci veya vuruşu yapan oyuncu ihlal etti mi?
- Top çizgiyi geçti mi?`
    },
    {
        name: "Law 12 - Sert Fauller ve Kartlar.txt",
        content: `IFAB FUTBOL OYUN KURALLARI - LAW 12: SERT FAULLER VE KARTLAR

TEDBİRSİZ MÜDAHALE (SARI KART)
==============================
Tedbirsiz müdahale, oyuncunun rakibinin güvenliğini tehlikeye attığı durumlarda yapılan müdahaledir.

SARI KART GEREKTIREN DURUMLAR:
• Oyunun akışını engelleyen faul (SPA - Stopping Promising Attack)
• Hakem kararına söz veya hareketle itiraz
• Rakibin umut verici bir hücumunu durdurma
• Provokatif, alaycı veya kışkırtıcı hareketler
• Oyunun yeniden başlatılmasını geciktirme
• Tekrar oyunda olmadan sahaya girme
• Kasıtlı olarak sahayı terk etme
• Korner/serbest vuruşlarda mesafeye uymama
• VAR inceleme alanına girme

AŞIRI GÜÇ KULLANIMI (KIRMIZI KART)
==================================
Ciddi Faul Oyunu (Serious Foul Play):
- Top için yapılan mücadelede aşırı güç kullanımı
- Vahşi müdahale
- Rakibin fiziksel bütünlüğünü tehlikeye atan hareketler

KIRMIZI KART GEREKTIREN DURUMLAR:
• Ciddi faul oyunu
• Şiddet içeren davranış (top mücadelesi dışında)
• Rakibe, hakeme veya başka birine tükürme
• Elleyle bariz gol şansını engelleme (kaleci kendi ceza sahasında hariç)
• DOGSO (bariz gol şansını faulle engelleme)
• Hakaret içeren, kaba sözler ve/veya hareketler
• İkinci sarı kart alma

KRAMPON GÖSTERME
================
Kayarak veya ayakta müdahalede tehlikeli durumlar:

TEHLİKELİ İŞARETLER:
• Kramponlar yukarı doğru
• Bacak düz ve sert (bükülmemiş)
• Diz kilitli
• Kontrol kaybı
• Yüksek hızda müdahale

Bu tür müdahaleler genellikle:
- SARI KART (tedbirsiz) veya
- KIRMIZI KART (aşırı güç) olarak değerlendirilir

VAR DEĞERLENDİRMESİ
===================
VAR şu durumlarda müdahale edebilir:
- Kırmızı kart gerektiren durum gözden kaçtıysa
- Yanlış oyuncuya kart gösterildiyse
- Ciddi faul oyunu değerlendirmesi

SPA (Umut Verici Atak) vs DOGSO
===============================
SPA (Stopping Promising Attack):
- Umut verici atak var ama bariz gol şansı değil
- SARI KART

DOGSO:
- Bariz gol şansı (4 kriter sağlanıyor)
- KIRMIZI KART (ceza sahası dışı) veya SARI KART (ceza sahası içi, topa oynama varsa)`
    },
    {
        name: "VAR Protokolü ve Minimum Müdahale.txt",
        content: `IFAB VAR PROTOKOLÜ - MİNİMUM MÜDAHALE İLKESİ

VAR NEDİR?
==========
Video Yardımcı Hakem (VAR), sahadaki hakeme yardımcı olan video sistemidir.
VAR, sadece "açık ve bariz hatalar" veya "gözden kaçan ciddi olaylar" için müdahale eder.

VAR MÜDAHALE EDEBİLECEĞİ DURUMLAR
=================================
1. GOL / GOL YOK
   - Topun çizgiyi geçip geçmediği
   - Gol öncesi ihlal olup olmadığı (faul, ofsayt, el)
   
2. PENALTI / PENALTI YOK
   - Faul var mı yok mu?
   - Ceza sahası içinde mi dışında mı?
   - El ile oynama ihlali var mı?
   
3. DOĞRUDAN KIRMIZI KART
   - Ciddi faul oyunu
   - Şiddet içeren davranış
   - Hakaret içeren davranış
   - DOGSO (ceza sahası dışı)
   
4. KİMLİK YANILGISI
   - Yanlış oyuncuya kart gösterilmesi

MİNİMUM MÜDAHALE PRENSİBİ
=========================
VAR sadece AÇlK VE BARİZ HATALAR için müdahale eder:

"Açık ve bariz" ne demek?
- Normal seyirci bile hatayı görebilir
- Tartışmalı, 50-50 durumlar VAR müdahalesi gerektirmez
- Hakemin yorumuna bırakılan durumlar değiştirilmez

VAR NE YAPAR?
- Kontrol eder (tüm gol ve penaltı kararları otomatik)
- Gerekirse hakemi uyarır
- Hakem OFR (On-Field Review) yapabilir

VAR PROTOKOLÜ ADIMLARI
======================
1. OLAY YAŞANIR
2. VAR otomatik kontrol başlatır
3. VAR "Kontrol ediyorum" der
4. Hakem oyunu bekletebilir
5. VAR analiz yapar
6. VAR sonucu iletir:
   - "Kontrol tamamlandı" (müdahale yok)
   - "Tavsiyem OFR" (hakem monitöre gitsin)
   - "Bariz hata var" (hakem doğrudan değiştirebilir)

OFR (ON-FIELD REVIEW)
=====================
Hakem sahaya kurulan monitörde görüntüleri izler:
- Öznel kararlar için (faul şiddeti, penaltı vs)
- Hakem son kararı kendisi verir
- VAR tavsiye eder, hakem karar verir

VAR SINIRLAMALARI
=================
VAR KULLANAMAZ/MÜDAHALE EDEMEZ:
- Sarı kart kararları (ikinci sarı dahil)
- Taç atışı yönü
- Korner/kale vuruşu kararları
- Oyunun genel akışı
- Faul olup olmadığı (penaltı ve kırmızı kart hariç)

OFSAYT HATASI
=============
- Yarı otomatik ofsayt (SAOT) kullanılabilir
- Milimetrik çizimler yapılır
- Vücut hatları kullanılır
- Kol ve eller sayılmaz`
    }
];

// Export for use
window.FootballRulesData = FootballRulesData;
