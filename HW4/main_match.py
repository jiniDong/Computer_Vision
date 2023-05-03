import hw_utils as utils
import matplotlib.pyplot as plt


def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('./data/scene', './data/book', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)

    # Test run matching with ransac

    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        './data/library', './data/library2',
        ratio_thres=0.5, orient_agreement=0.15, scale_agreement=0.25)
    plt.title('MatchRANSAC')
    plt.imshow(im)




if __name__ == '__main__':
    main()
