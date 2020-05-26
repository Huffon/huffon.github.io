---
layout: post
title: "TORCH.UTILS.DATA 공식 문서 파훼하기"
subtitle: '데이터 피딩에 활용되는 PyTorch 클래스를 알아보자'
author: "devfon"
header-style: text
lang: kr
tags:
  - AI
  - Deep Learning
---

PyTorch **데이터 로딩**의 중심에는 [`torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) 클래스가 있습니다. **DataLoader**는 **Dataset**에 대한 **Python Iterable** 클래스입니다. **DataLoader**에는 다양한 옵션이 존재하는데, 여러 옵션을 활용해 다음과 같이 **DataLoader**를 초기화 할 수 있습니다.

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
```

이제 **DataLoader** 클래스 초기화에 활용되는 **다양한 옵션**들에 대해 알아보도록 하겠습니다.

<br/>

## Dataset 종류

`DataLoader` 초기화에 있어 가장 중요한 인자는 `dataset` 이며, 이는 데이터를 끌어올 **데이터셋 객체**를 지칭합니다. PyTorch는 **두 개** 유형의 데이터셋을 지원합니다.

### Map-style 데이터셋

**Map-style 데이터셋**은 `__getitem__()`과 `__len__()`을 구현해 인덱스, 키 등을 활용한 **데이터 샘플 매핑**을 수행합니다. 예를 들어, **Map-style 데이터셋**은 `dataset[idx]`와 같이 접근되었을 때, `idx`번 째 **이미지**와 해당 이미지의 **라벨**을 매핑해 반환하게 됩니다.

<br/>

### Iterable-style 데이터셋

**Iterable-style 데이터셋**은 `__iter__()`를 구현하는 [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)의 서브 클래스 인스턴스입니다. 이는 데이터 샘플에 대한 **Iterable** 객체입니다. 해당 데이터셋은 데이터 샘플을 임의로 읽는 작업이 **비싼 연산**이거나 **부적절**할 때, 혹은 **배치 사이즈**가 `__iter__()` 연산을 통해 읽어온 데이터 갯수에 의해 정해지게끔 하고 싶을 때 사용하기 적합한 데이터셋입니다.

예를 들어, `iter(dataset)`이 호출되었을 때, 데이터셋은 **데이터베이스**, **원격 서버**, 혹은 실시간으로 생성되는 **로그** 등에서 **데이터 스트림**을 읽어와 반환합니다.

_Note: `IterableDataset`을 멀티 프로세스로 활용하면, **동일한 데이터셋 객체**가 **각 워커 프로세스에 복제**됩니다. 따라서 **중복 데이터**가 모델에 **Feeding** 되지 않도록 하기 위해서는 추가 설정이 수행되어야 합니다._

```python
# 중복 데이터 방지하는 예시

def __iter__(self):
	worker_info = torch.utils.data.get_worker_info()
	if worker_info is None:  # 싱글 프로세스로 데이터 로딩할 경우, Full Iterator를 반환
		iter_start = self.start
		iter_end = self.end
	else:  # 멀티 프로세스로 데이터 로딩할 경우, 워크 로드 분배
		per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
		worker_id = worker_info.id
		iter_start = self.start + worker_id * per_worker
		iter_end = min(iter_start + per_worker, self.end)
	return iter(range(iter_start, iter_end))
```

<br/>

## 데이터 로딩 순서와 [Sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)

**Iterable-style 데이터셋**의 데이터 로딩 순서는 전적으로 **사용자가 정의한 Iterable**에 의해 결정됩니다. 그리고 이러한 특성은 **Chunk 단위**로 데이터를 읽는다거나, **동적인 배치 사이즈**를 설정한다거나 등의 작업을 구현하기 쉽게 도와줍니다. 단순히 **Iterable**이 **Batch 단위**의 샘플을 `yield`만 하면 되기 때문입니다.

따라서 이번 섹션에서는 **Map-style 데이터셋**만 다룰 예정입니다. `Sampler`는 **Map-style 데이터셋**의 데이터 로딩에 사용되는 **인덱스 혹은 키 시퀀스**를 정하기 위해 사용되는 클래스입니다. 즉, **Sampler** 클래스는 **데이터셋에 접근할 수 있는 인덱스**에 대한 **Iterable 객체**입니다. 예를 들어, **Stochastic Gradient Decent (SGD)**에 있어 **Sampler**는 인덱스로 구성된 시퀀스를 **순열**한 후, 매번 한 개의 인덱스를 `yield` 합니다. 혹은 **미니 배치 SGD**를 위해 **특정 갯수**만큼의 인덱스를 `yield` 할 수도 있습니다.

**Sequential** 혹은 **Shuffled Sampler**의 경우, `DataLoader` 인스턴스 초기화 옵션에 사용되는 `shuffle` 인자에 따라 결정 및 구현됩니다. 혹은 사용자가 직접 구현한대로 인덱스 및 키 시퀀스를 `yield` 해주는 **커스텀 Sampler**를 `sampler` 인자에 넣어줄 수도 있습니다.

**커스텀 Sampler**가 배치 단위로 인덱스를 `yield` 하도록 설정하고자 한다면, `batch_sampler` 인자에 **커스텀 Sampler**를 넘겨주면 됩니다. 혹은 단순 배치만을 위해 **커스텀 Sampler**를 구현하고자 하는 것이라면, 이는 `batch_size`와 `drop_last` 인자를 통해 해당 설정이 가능합니다.

_역자주: `drop_last` 인자의 경우, 배치 크기를 채우지 못한 **마지막 불완전 배치**를 사용할 것인지, 사용하지 않을 것인지를 결정하는 **Boolean** 인자입니다._

_Note: **Iterable-style 데이터셋**의 경우, 인덱스 혹은 키 등의 개념이 없기 때문에 `sampler`와 `batch_sampler` 인자를 사용할 수 없습니다._

<br/>

## 배치 혹은 Non-배치 데이터의 로딩

**DataLoader** 클래스는 `batch_size`, `drop_last` 그리고 `batched_sampler` 인자를 통해 개별 데이터 샘플을 **배치**로 엮어주는 기능을 제공합니다.

### 자동 배치 (기본 값)

**자동 배치**는 가장 흔히 사용되는 옵션입니다. **자동 배치**의 경우, **미니 배치** 크기만큼의 데이터를 읽어와 이를 **배치 샘플**로 합쳐줍니다. 때문에 **배치 샘플 텐서**의 차원 하나는 **배치 사이즈**를 나타내며, 주로 **첫 번째 차원**이 이를 나타냅니다.

`batch_size`가 `None`이 아니라면, **DataLoader**는 **배치 샘플**을 `yield`하게 됩니다. 앞서 설명한 것과 마찬가지로, **Map-style 데이터셋**의 경우, 인덱스 시퀀스를 `yield` 해주는 **Sampler**를 구현해 `batch_sampler` 인자에 해당 **Sampler** 객체를 넘겨주는 방식으로도 배치 샘플을 모델에 **Feeding** 할 수 있습니다.

_Note: `batch_size`와 `drop_last` 인자는 `sampler`로부터 `batch_sampler`를 구성하기 위해 사용됩니다. **Map-style 데이터셋**의 경우, `sampler`가 사용자에 의해 전달되거나, `shuffle` 인자에 의해 결정됩니다. **Iterable-style 데이터셋**의 경우 `sampler`에 더미 값 `1`이 들어가게 됩니다._

_Note: **Iterable-style 데이터셋**을 멀티 프로세서로 활용한다면, `drop__last` 인자를 통해 각 워커가 복제해 사용하는 데이터셋에서 **마지막 미완성 배치**를 활용할지 말지 정하게 됩니다._

**Sampler**가 내놓은 인덱스 시퀀스를 활용해 데이터 샘플을 읽어온 후에는 `collate_fn` 인자로 등록된 **함수**가 **샘플 리스트**를 **배치 샘플**로 합치는데 활용됩니다.

따라서 **Map-style 데이터셋**에서 데이터를 읽어오는 과정은 다음과 같아집니다:

```python
for indices in batch_sampler:
	yield collate_fn([dataset[i] for i in indices])
```

그리고 **Iterable-style 데이터셋**에서 데이터를 읽어오는 과정은 다음과 같아집니다:

```python
dataset_iter = iter(dataset)
for indices in batch_sampler:
	yield collate_fn([next(dataset_iter) for _ in indices])
```

추가적으로 배치 내 존재하는 **시퀀셜 데이터**를 **최대 길이만큼 패딩**한다거나 등의 **추가 작업**을 수행할 수 있는 **커스텀 `collate_fn`**을 구현해 활용할 수도 있습니다.

<br/>

### 자동 배치 해제

특수한 경우에 사용자는 데이터셋 코드에 있어 **배치를 직접 관리**하거나, **개별 데이터 샘플**을 읽어들여야 할 수도 있습니다. 예를 들어, 때로는 데이터를 데이터베이스에서 직접 **Bulk**로 읽거나, 메모리에서 연속 **Chunk**로 읽어오는 것이 (즉, 이미 **배치 상태**인 데이터를 바로 읽어오는 것이) 더 저렴한 연산일 수 있습니다. 혹은 배치 사이즈가 **데이터에 따라 다르게 적용**되어야 하는 경우도 있을 것입니다. 이러한 경우, 자동 배치를 해제해 **DataLoader**가 `dataset` 객체의 **각 샘플**을 반환하게끔 하는 것이 좋습니다.

`batch_size`와 `batch_sampler`가 모두 `None` 일 때, **자동 배치**가 해제됩니다. 그리고 `dataset`을 통해 얻어진 개별 데이터 샘플은 `collate_fn` 함수를 거쳐 모델에 **Feeding** 됩니다.

**자동 배치 설정이 해제**된 경우, 기본 `collate_fn`이 수행하는 작업은 단순히 **NumPy 배열**을 **PyTorch 텐서**로 **컨버팅**해주는 것입니다.

이 경우, **Map-style 데이터셋**에서 데이터를 읽어오는 과정이 다음과 같아집니다:

```python
for index in sampler:
	yield collate_fn(dataset[index])
```

그리고 **Iterable-style 데이터셋**에서 데이터를 읽어오는 과정은 다음과 같아집니다:

```python
for data in iter(dataset):
	yield collate_fn(data)
```

<br/>

### `collate_fn` 활용하기

`collate_fn`은 **자동 배치** 설정에 따라 다르게 적용됩니다.

**자동 배치가 해제**된 경우, `collate_fn`은 **개별 데이터 샘플에 대해** 적용됩니다. 이때의 `collate_fn`은 앞서 언급했듯 단순히 **NumPy 배열**을 **PyTorch 텐서**로 변환하는 작업을 수행하게 됩니다.

**자동 배치가 설정**된 경우, `collate_fn`은 **데이터 샘플 리스트**에 대해 적용됩니다. 이때의 `collate_fn`은 리스트에 포함되어 있는 **데이터 샘플들을 배치 샘플로 합치는 작업**을 수행합니다.

예를 들어, 각 데이터 샘플이 **3 채널 이미지**와 **정수 클래스 라벨**로 구성되어 `dataset`의 반환 값이 `(image, class_index)`의 튜플인 경우, 기본 `collate_fn`은 해당 튜플로 구성된 리스트를 하나의 튜플로 합쳐 **배치 이미지 텐서**와 **배치 클래스 라벨 텐서**를 생성합니다. 그리고 특히, 기본 `collate_fn`은 다음과 같은 특성을 지닙니다:

- 항상 텐서 가장 앞에 **배치 크기** 차원을 추가합니다.
- **NumPy 배열**과 **Python 수치 값**을 **PyTorch 텐서**로 컨버팅합니다.
- `list`, `tuple`, `dictionary`, `namedtuple` 등 입력 **자료형의 구조를 보존**합니다. 예를 들어, 각 데이터 샘플이 딕셔너리였을 경우, 결과 역시 동일한 키와 값으로 구성된 딕셔너리로 나오게 됩니다. 다만, 키에 상응하는 값은 **배치 텐서**로 변환되어 반환됩니다.

또한 사용자는 배치 사이즈를 첫 번째 차원으로 사용하지 않게끔 한다거나, 시퀀셜 데이터에 **패딩을 적용**한다거나 등으로 `collate_fn`을 커스터마이즈해 활용할 수도 있습니다.