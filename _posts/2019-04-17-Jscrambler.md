---
layout: post
title: "포르투갈 IT 기업을 알아보자: 'Jscrambler'"
subtitle: 'Portugal에서 Netflix를 고객으로 삼는 개발사가 있다고?'
author: "devfon"
header-style: text
lang: kr
tags:
  - Portugal
  - IT
---

![](/img/in-post/javascripts.jpg =300x)

**JavaScript**는 웹 개발을 경험해 본 적이 없는 사람이라도, 웹 서핑을 하며 브라우져에서 많이 접해본 단어일 것이다. 이 JavaScript는 웹 개발, 그 중에서도 웹 브라우져의 화면에 보이는 부분을 담당하는 프로그래밍 언어이다. 즉, Front-end에서 사용되는 기능들을 설계하기 위해 사용되는 언어인 것이다.

Web 2.0의 도래 이래로 웹 페이지를 구현하기 위한 기술은 점점 더 고도화되어가고 있으며, 이러한 흐름에 맞춰 다양한 웹 솔루션들 역시 우후죽순으로 쏟아져 나오고 있다. 그리고 이에 따라 웹 개발에서 주력으로 사용되는 언어인 JavaScript를 기반으로 한 오픈 소스나 솔루션 역시 엄청나게 쏟아지고 있다.

![](/img/in-post/jsgit.png)
> 코드 저장소 Github에서의 프로그래밍 언어 인기도

포르투갈 기업 이야기를 한다더니 왜 프로그래밍 언어에 대한 설명만 하는지 의아할 수 있을 것이다. 포르투갈 기업 이야기에 앞서 JavaScript에 대한 설명을 한 것은 포르투갈에서 JavaScript 기반의 웹 솔루션을 개발해 좋은 성과를 거두고 있는 기업이 있기 때문이다.

![](/img/in-post/jscrambler.png)
지금부터 소개할 **Jscrambler**는 JavaScript 기반의 웹 보안 솔루션을 개발한 포르투갈의 소프트웨어 기업이다. 이 기업은 웹 사이트의 절대적인 수가 급증하고, 그로 인한 웹 페이지 광고 시장이 도래함에 따라 발전한 신종 사기 기법 **Click fraud advertising campaign**(클릭 수를 늘리기 위한 광고)을 제지하기 위한 아이디어 고안에서 시작되었다.

해당 사기는 보통 '클릭 당 광고 비용 지불'을 제공하는 웹 페이지 광고사에서 주로 발생하는데, 사용자가 웹 페이지에 떠있는 광고를 보고 직접 클릭을 하지 않더라도 웹 페이지의 JavaScript 코드를 살짝 건드려 사용자의 브라우져에서 자동으로 해당 광고를 클릭을 하게끔 만드는 기술이다.

이러한 사기 기법을 방지하기 위해서는 페이지를 구성하고 있는 JavaScript 코드를 보안해야 하는데, Jscrambler가 당시 이 아이디어를 고안하였을 때만 해도 JavaScript 보안 솔루션이 시장에 존재하지 않아 직접 시장에 뛰어들게 되었다고 한다.

#### The Technology of Jscrambler
![](/img/in-post/jscrambler.png)

Jscrambler의 대부분의 솔루션 제품은 JavaScript 코드를 보호해주는 데에 초점을 맞추고 있다. 위 그림과 같이 웹 페이지를 구성하고 있는 JavaScript 코드를 악성 코드나 버그 등이 해칠 수 없도록 페이지 소스 코드를 감싸 보호해주는 솔루션을 제공하는 것이다.

![](/img/in-post/jstech.png)

또한 Jscrambler는 단순한 JavaScript 보안 솔루션에서 더 나아가 사용자 측의 웹 페이지. 즉, Client-side의 보안까지 담당하고 있다고 한다. 따라서 사용자가 보안에 취약한 환경에서 웹 페이지를 사용할 때 사,용자의 접속 환경을 모니터링하여 안전하게 웹 페이지를 이용할 수 있도록 보호해주는 기능을 수행하게 된다.

#### Startup Plan
![](/img/in-post/jssp.png)

추가적으로 Jscrambler를 알아보며 조금 재밌는 요금제를 발견해 소개하며 글을 마치고자 한다. 앞서 여러 포스팅을 통해 포르투갈에서의 스타트업에 대한 이야기를 한 적이 있었다. 해당 글들을 작성하기 위해 자료 조사를 해보며 포르투갈 내의 스타트업 문화가 상당히 건전하게 잘 형성되어 있다고 생각을 했는데, Jscrambler도 그러한 건전한 스타트업 문화를 보여주고 있었다.

Jscrambler는 이제 꽤나 명성을 지닌 웹 솔루션 기업이 되어 솔루션 비용이 꽤나 비싸다 *(사실 정확히 공개되어 있지는 않지만 특별 요금제인 Startup Plan만 봐도 비쌀 것이다)*. 그리고 이들은 자신과 같이 테크 스타트업으로 사업을 시작하는 것에 대한 고충을 알기 때문인지(?) 스타트업 기업들에게 솔루션을 훨씬 저렴한 가격에 이용 가능하게 해주는 Startup Plan을 제공하고 있다. 

해당 Startup Plan을 이용하기 위해서는 현재 자신의 스타트업이 시드 머니를 가진 상태이거나, Series A 상태에 있어야 하며, 해당 조건을 맞춘 이후에는 Jscrambler 측과의 상담을 통해 해당 요금제를 적용 받을 수 있다고 한다.

#### References
[Jscrambler](https://jscrambler.com/)

[Programming Language ranking](https://hackernoon.com/top-3-most-popular-programming-languages-in-2018-and-their-annual-salaries-51b4a7354e06)