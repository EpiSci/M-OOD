## Mixture_Density_Network

MDN을 이용하여, explainable variance와 unexplainable variance을 구하여, Uncertainty를 구합니다.

최성준 박사님의 tensorflow로 구현된 코드를 keras로 구현하였습니다.

## MDN

![mdn](http://edwardlib.org/images/mixture-density-network-fig0.png)

위의 이미지와 같이 multimodal한 data를 fitting 시킬 수 있다.

특히 , MDN은 single output을 predict하는 것이 아니라, output의 probabiltiy distribution을 predict한다.

## Usage

```bash
python3 demo_mdn.py # training and visualization
```

## Precautions

### unstable loss

var가 0으로 수렴하면 올바른 distribution을 생성할 수 없게 됩니다. 이는 unstable한 loss를 생성하게 됩니다.

이러한 문제를 해결하기 위해서 var에 small positive value를 더하여 일정수준 이상의 var를 가지도록 하였습니다.

### underfitting issue

batch_normalization을 이용하여 개선하였습니다.

## result

### fitting result

![result](https://github.com/RRoundTable/Mixture_Density_Network/raw/master/result/result.gif)

- 0 epoch

![epoch0](https://github.com/RRoundTable/Mixture_Density_Network/raw/master/result/epoch_0.png

- 16000 epoch

![epoch_16000](https://github.com/RRoundTable/Mixture_Density_Network/raw/master/result/epoch_16000.png)



### explainable/ unexplainable variance

expainable variance란 training data를 더 수집하면 개선할 수 있는 uncertainty를 의미한다.

반면에 unexplainable variance는 data자체의 noise로 training data를 더 수집하여도 개선할 수 없다.

![variance](https://github.com/RRoundTable/Mixture_Density_Network/raw/master/variance/variance.gif)

- 0 epoch

![epoch0](https://github.com/RRoundTable/Mixture_Density_Network/raw/master/variance/epoch_0.png)

- 16000 epoch

![epoch_16000](https://github.com/RRoundTable/Mixture_Density_Network/raw/master/variance/epoch_16000.png)


## reference

- github : https://github.com/sjchoi86/density_network