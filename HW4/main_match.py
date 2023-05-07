import hw_utils as utils
import matplotlib.pyplot as plt


import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/box', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)

    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library2', './data/library',
        ratio_thres=0.75, orient_agreement=45, scale_agreement=0.80)
    plt.title('MatchRANSAC')
    plt.imshow(im)

if __name__ == '__main__':
    main()